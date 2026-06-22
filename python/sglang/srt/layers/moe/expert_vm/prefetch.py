from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.layers.moe.expert_vm.config import get_expert_vm_config
from sglang.srt.layers.moe.expert_vm.gather import (
    allocate_compact_gpu_tensors,
    gather_expert_rows_async,
    get_active_expert_ids,
    remap_topk_ids,
)
from sglang.srt.layers.moe.expert_vm.weights import get_expert_vm_param_names
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput, TopKOutputChecker

if TYPE_CHECKING:
    from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock

logger = logging.getLogger(__name__)


@dataclass
class _PrefetchJob:
    layer_id: int
    active_ids: torch.Tensor
    gpu_tensors: Dict[str, torch.Tensor]
    copy_start_event: torch.cuda.Event
    copy_end_event: torch.cuda.Event
    bytes_transferred: int
    # topk_output to bind for this layer's MoE compute.  Always matches the
    # expert rows actually staged into VRAM (predicted for lookahead jobs,
    # actual for cold-start jobs), so dispatch never references an unloaded expert.
    bind_topk: Optional[StandardTopKOutput] = None
    is_lookahead: bool = False


@dataclass
class _LayerRuntime:
    fused_moe: FusedMoE
    pending_job: Optional[_PrefetchJob] = None
    bound_job: Optional[_PrefetchJob] = None
    remapped_topk: Optional[StandardTopKOutput] = None


class ExpertVMPrefetchCoordinator:
    """Async top-k expert staging from CPU RAM to compact GPU buffers."""

    def __init__(self) -> None:
        self._sparse_blocks: Dict[int, "Qwen3MoeSparseMoeBlock"] = {}
        self._layers: Dict[int, _LayerRuntime] = {}
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._stats_prefetches: int = 0
        self._stats_lookaheads: int = 0

    def register_sparse_block(self, block: "Qwen3MoeSparseMoeBlock") -> None:
        layer_id = block.layer_id
        self._sparse_blocks[layer_id] = block
        self._layers[layer_id] = _LayerRuntime(fused_moe=block.experts)

    def _get_copy_stream(self) -> torch.cuda.Stream:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(priority=-1)
        return self._copy_stream

    def _is_resident(self, layer_id: int) -> bool:
        cfg = get_expert_vm_config()
        return cfg is not None and cfg.is_resident_layer(layer_id)

    def _should_stage(self, layer: FusedMoE) -> bool:
        cfg = get_expert_vm_config()
        if cfg is None or not getattr(layer, "_expert_vm_offloaded", False):
            return False
        return not cfg.is_resident_layer(layer.layer_id)

    @staticmethod
    def _normalize_topk(
        topk_output: TopKOutput, layer_id: int
    ) -> StandardTopKOutput:
        if TopKOutputChecker.format_is_standard(topk_output):
            return topk_output
        return topk_output.to_standard(layer_id=layer_id)

    def _gather_and_submit_dma(
        self,
        layer: FusedMoE,
        layer_id: int,
        param_names: List[str],
        active_ids: torch.Tensor,
        k: int,
        gpu_tensors: Dict[str, torch.Tensor],
        copy_stream: torch.cuda.Stream,
        copy_start: torch.cuda.Event,
        copy_end: torch.cuda.Event,
        job: _PrefetchJob,
    ) -> None:
        # Single tiny D2H to learn which expert rows to copy.
        if active_ids.device.type != "cpu":
            ids_list = active_ids.to(dtype=torch.int64, device="cpu").tolist()
        else:
            ids_list = active_ids.to(dtype=torch.int64).tolist()

        # Direct row-by-row DMA from PINNED cpu_buf -> compact GPU tensor.
        # No CPU index_select gather: the copy engine reads pinned host rows
        # directly and the copies are truly non-blocking, overlapping compute.
        bytes_xferred = 0
        with torch.cuda.stream(copy_stream):
            copy_start.record(copy_stream)
            for name in param_names:
                cpu_buf = getattr(layer, f"expert_vm_{name}_cpu", None)
                gpu_t = gpu_tensors.get(name)
                if cpu_buf is None or gpu_t is None:
                    continue
                gather_expert_rows_async(cpu_buf, ids_list, gpu_t, copy_stream)
                bytes_xferred += gpu_t.numel() * gpu_t.element_size()
            copy_end.record(copy_stream)

        job.bytes_transferred = bytes_xferred

    def begin_prefetch(
        self,
        layer: FusedMoE,
        topk_output: TopKOutput,
    ) -> None:
        """Prefetch layer L's experts synchronously on the main thread.

        If a lookahead job already exists (pre-fetched by the previous layer's
        begin_lookahead), this is a no-op — wait_and_bind handles the hit/miss check.

        Cold-start (first layer, no prior lookahead): gather + DMA run here, then
        wait_and_bind uses wait_event (GPU-side) so CPU returns immediately.
        """
        if not self._should_stage(layer):
            return

        layer_id = layer.layer_id
        runtime = self._layers[layer_id]

        if runtime.pending_job is not None:
            return  # lookahead already submitted this layer's DMA

        # Cold-start: compute active experts and submit DMA synchronously.
        std_topk = self._normalize_topk(topk_output, layer_id)
        active_ids, k = get_active_expert_ids(std_topk.topk_ids)
        param_names = get_expert_vm_param_names(layer)
        device = torch.device("cuda", torch.cuda.current_device())
        copy_stream = self._get_copy_stream()
        copy_start = torch.cuda.Event(enable_timing=False)
        copy_end = torch.cuda.Event(enable_timing=False)

        gpu_tensors = allocate_compact_gpu_tensors(layer, param_names, active_ids, device)
        job = _PrefetchJob(
            layer_id=layer_id,
            active_ids=active_ids,
            gpu_tensors=gpu_tensors,
            copy_start_event=copy_start,
            copy_end_event=copy_end,
            bytes_transferred=0,
            bind_topk=std_topk,  # cold start: actual topk is exact
            is_lookahead=False,
        )
        runtime.pending_job = job
        self._gather_and_submit_dma(
            layer, layer_id, param_names, active_ids, k,
            gpu_tensors, copy_stream, copy_start, copy_end, job,
        )
        self._stats_prefetches += 1

    def begin_lookahead_prefetch_during_compute(
        self,
        current_layer: FusedMoE,
        hidden_states: torch.Tensor,
    ) -> None:
        """Predict and prefetch layer L+1's experts (called between L's router and moe).

        Cross-layer prediction: gate+topk for L+1 is computed from layer L's hidden
        states, then those expert rows are copied CPU->GPU on the copy_stream.

        The only CPU<->GPU sync here is a single tiny D2H of the predicted expert ids
        (needed to index the CPU-resident weight buffers).  The DMA itself is issued
        non-blocking and overlaps layer L's MoE compute; layer L+1's wait_and_bind
        later does a GPU-side wait_event with no CPU stall.
        """
        if not self._should_stage(current_layer):
            return

        next_id = current_layer.layer_id + 1
        next_block = self._sparse_blocks.get(next_id)
        if next_block is None or self._is_resident(next_id):
            return
        if not getattr(next_block.experts, "_expert_vm_offloaded", False):
            return

        next_runtime = self._layers[next_id]
        if next_runtime.pending_job is not None:
            return  # already have a job for next layer

        layer = next_block.experts
        param_names = get_expert_vm_param_names(layer)
        copy_stream = self._get_copy_stream()
        device = torch.device("cuda", torch.cuda.current_device())

        copy_start = torch.cuda.Event(enable_timing=False)
        copy_end = torch.cuda.Event(enable_timing=False)

        # Predict L+1's experts from L's hidden states (cross-layer gate prediction).
        with torch.no_grad():
            router_logits, _ = next_block.gate(hidden_states)
            topk_output = next_block.topk(hidden_states, router_logits)

        std_topk = self._normalize_topk(topk_output, next_id)
        active_ids, k = get_active_expert_ids(std_topk.topk_ids)

        gpu_tensors = allocate_compact_gpu_tensors(layer, param_names, active_ids, device)

        job = _PrefetchJob(
            layer_id=next_id,
            active_ids=active_ids,
            gpu_tensors=gpu_tensors,
            copy_start_event=copy_start,
            copy_end_event=copy_end,
            bytes_transferred=0,
            bind_topk=std_topk,  # compute L+1 with the experts we staged
            is_lookahead=True,
        )
        next_runtime.pending_job = job

        # Gather + DMA (single D2H of ids inside; copy is async on copy_stream).
        self._gather_and_submit_dma(
            layer, next_id, param_names, active_ids, k,
            gpu_tensors, copy_stream, copy_start, copy_end, job,
        )
        self._stats_lookaheads += 1

    def wait_and_bind(
        self, layer: FusedMoE, topk_output: TopKOutput
    ) -> TopKOutput:
        if not self._should_stage(layer):
            return topk_output

        layer_id = layer.layer_id
        runtime = self._layers[layer_id]

        job = runtime.pending_job
        if job is None:
            # Cold start (e.g. first offloaded layer): synchronous prefetch.
            std_topk = self._normalize_topk(topk_output, layer_id)
            self.begin_prefetch(layer, std_topk)
            job = runtime.pending_job

        assert job is not None and job.bind_topk is not None

        # GPU-side dependency only: the compute stream waits for the H2D copy to
        # finish before dispatch/GEMM.  The CPU returns immediately — no stall, and
        # no D2H readback (we always bind the topk matching the staged experts).
        torch.cuda.current_stream().wait_event(job.copy_end_event)

        bind_topk = job.bind_topk
        remapped_ids = remap_topk_ids(
            bind_topk.topk_ids, job.active_ids, layer.num_experts
        )
        remapped_topk = StandardTopKOutput(
            bind_topk.topk_weights, remapped_ids, bind_topk.router_logits
        )

        param_names = get_expert_vm_param_names(layer)
        for name in param_names:
            gpu_t = job.gpu_tensors.get(name)
            if gpu_t is None:
                continue
            param = getattr(layer, name)
            param.data = gpu_t

        runtime.bound_job = job
        runtime.pending_job = None
        runtime.remapped_topk = remapped_topk
        return remapped_topk

    def release(self, layer: FusedMoE) -> None:
        if not self._should_stage(layer):
            return

        layer_id = layer.layer_id
        runtime = self._layers.get(layer_id)
        if runtime is None or runtime.bound_job is None:
            return

        device = torch.device("cuda", torch.cuda.current_device())
        param_names = get_expert_vm_param_names(layer)
        for name in param_names:
            if not hasattr(layer, name):
                continue
            param = getattr(layer, name)
            param.data = torch.empty(0, device=device, dtype=param.dtype)

        runtime.bound_job = None
        runtime.remapped_topk = None


_coordinator: Optional[ExpertVMPrefetchCoordinator] = None


def get_expert_vm_coordinator() -> ExpertVMPrefetchCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = ExpertVMPrefetchCoordinator()
    return _coordinator


def reset_expert_vm_coordinator() -> None:
    global _coordinator
    _coordinator = None


def register_expert_vm_sparse_blocks(model: torch.nn.Module) -> None:
    from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    coord = get_expert_vm_coordinator()
    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3MoeSparseMoeBlock):
            coord.register_sparse_block(module)
            count += 1
    # logger.info("[expert_vm] Registered %d sparse MoE blocks for prefetch.", count)


def expert_vm_begin_prefetch(layer: FusedMoE, topk_output: TopKOutput) -> None:
    get_expert_vm_coordinator().begin_prefetch(layer, topk_output)


def expert_vm_begin_lookahead_during_compute(
    layer: FusedMoE, hidden_states: torch.Tensor
) -> None:
    get_expert_vm_coordinator().begin_lookahead_prefetch_during_compute(
        layer, hidden_states
    )


# Public alias used by qwen3_moe.py between router and moe steps.
expert_vm_begin_lookahead = expert_vm_begin_lookahead_during_compute


def expert_vm_wait_and_bind(
    layer: FusedMoE, topk_output: TopKOutput
) -> TopKOutput:
    return get_expert_vm_coordinator().wait_and_bind(layer, topk_output)


def expert_vm_release(layer: FusedMoE) -> None:
    get_expert_vm_coordinator().release(layer)
