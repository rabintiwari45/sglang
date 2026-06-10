from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.layers.moe.expert_vm.config import get_expert_vm_config
from sglang.srt.layers.moe.expert_vm.gather import (
    allocate_compact_gpu_tensors,
    expert_ids_as_sorted_list,
    expert_set_lookahead_diff,
    expert_sets_match,
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
    is_lookahead: bool = False
    lookahead_topk: Optional[StandardTopKOutput] = None
    use_predicted_topk: bool = False


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
        self._stats_lookahead_hits: int = 0
        self._stats_lookahead_misses: int = 0

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

    def begin_prefetch(
        self,
        layer: FusedMoE,
        topk_output: TopKOutput,
        *,
        is_lookahead: bool = False,
    ) -> None:
        if not self._should_stage(layer):
            return

        layer_id = layer.layer_id
        std_topk = self._normalize_topk(topk_output, layer_id)
        active_ids, k = get_active_expert_ids(std_topk.topk_ids)
        runtime = self._layers[layer_id]

        # If a lookahead is already pending for this layer with the same expert
        # set, reuse it — avoid a redundant H2D copy when forward_normal fires
        # its own begin_prefetch right after the lookahead transfer has already
        # started on the copy stream.
        existing = runtime.pending_job
        if existing is not None and existing.is_lookahead and not is_lookahead:
            predicted, actual, wrong_prefetch, missing, overlap = (
                expert_set_lookahead_diff(existing.active_ids, active_ids)
            )
            if expert_sets_match(active_ids, existing.active_ids):
                existing.is_lookahead = False  # promote to confirmed prefetch
                self._stats_prefetches += 1
                self._stats_lookahead_hits += 1
                logger.info(
                    "[expert_vm] Lookahead hit layer=%d | predicted=%s actual=%s",
                    layer_id,
                    predicted,
                    actual,
                )
                return

            # Miss: keep predicted weights; route with predicted topk so compact
            # indices stay in range (actual topk would remap to -1 and crash MoE).
            self._stats_lookahead_misses += 1
            logger.info(
                "[expert_vm] Lookahead miss layer=%d | "
                "predicted=%s actual=%s | "
                "wrong_prefetch=%s missing=%s overlap=%s | "
                "using predicted topk for compute",
                layer_id,
                predicted,
                actual,
                wrong_prefetch,
                missing,
                overlap,
            )
            existing.is_lookahead = False
            existing.use_predicted_topk = True
            return

        if not is_lookahead and (existing is None or not existing.is_lookahead):
            logger.info(
                "[expert_vm] Prefetch actual layer=%d experts=%s (no prior lookahead)",
                layer_id,
                expert_ids_as_sorted_list(active_ids),
            )

        param_names = get_expert_vm_param_names(layer)
        device = torch.device("cuda", torch.cuda.current_device())

        gpu_tensors = allocate_compact_gpu_tensors(
            layer, param_names, active_ids, device
        )
        bytes_xferred = 0
        copy_stream = self._get_copy_stream()
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(copy_stream):
            copy_start.record(copy_stream)
            for name in param_names:
                cpu_buf = getattr(layer, f"expert_vm_{name}_cpu", None)
                gpu_t = gpu_tensors.get(name)
                if cpu_buf is None or gpu_t is None:
                    continue
                gather_expert_rows_async(cpu_buf, active_ids, gpu_t, copy_stream)
                bytes_xferred += gpu_t.numel() * gpu_t.element_size()
            copy_end.record(copy_stream)

        job = _PrefetchJob(
            layer_id=layer_id,
            active_ids=active_ids,
            gpu_tensors=gpu_tensors,
            copy_start_event=copy_start,
            copy_end_event=copy_end,
            bytes_transferred=bytes_xferred,
            is_lookahead=is_lookahead,
            lookahead_topk=std_topk if is_lookahead else None,
        )
        runtime.pending_job = job

        if is_lookahead:
            logger.info(
                "[expert_vm] Lookahead predicted layer=%d experts=%s",
                layer_id,
                expert_ids_as_sorted_list(active_ids),
            )

        # kind = "lookahead" if is_lookahead else "prefetch"
        # logger.info(
        #     "[expert_vm] %s started layer=%d active_experts=%d bytes=%.2f MiB",
        #     kind,
        #     layer_id,
        #     k,
        #     bytes_xferred / (1024**2),
        # )
        self._stats_prefetches += 0 if is_lookahead else 1
        self._stats_lookaheads += 1 if is_lookahead else 0

    def begin_lookahead_prefetch_during_compute(
        self,
        current_layer: FusedMoE,
        hidden_states: torch.Tensor,
    ) -> None:
        """While layer L experts compute, speculatively prefetch layer L+1 top-k experts."""
        if not self._should_stage(current_layer):
            return

        next_id = current_layer.layer_id + 1
        next_block = self._sparse_blocks.get(next_id)
        if next_block is None or self._is_resident(next_id):
            return
        if not getattr(next_block.experts, "_expert_vm_offloaded", False):
            return

        with torch.no_grad():
            router_logits, _ = next_block.gate(hidden_states)
            topk_output = next_block.topk(hidden_states, router_logits)

        # logger.info(
        #     "[expert_vm] Lookahead gate+topk for layer=%d during layer=%d expert compute",
        #     next_id,
        #     current_layer.layer_id,
        # )
        self.begin_prefetch(
            next_block.experts, topk_output, is_lookahead=True
        )

    def wait_and_bind(
        self, layer: FusedMoE, topk_output: TopKOutput
    ) -> TopKOutput:
        if not self._should_stage(layer):
            return topk_output

        layer_id = layer.layer_id
        std_topk = self._normalize_topk(topk_output, layer_id)
        active_ids, k = get_active_expert_ids(std_topk.topk_ids)
        runtime = self._layers[layer_id]

        job = runtime.pending_job
        if job is None:
            # No lookahead from prior layer (e.g. layer 1 after resident layer 0).
            self.begin_prefetch(layer, std_topk, is_lookahead=False)
            job = runtime.pending_job

        assert job is not None
        t0 = time.perf_counter()
        job.copy_end_event.synchronize()
        wait_ms = (time.perf_counter() - t0) * 1000
        if job.bytes_transferred > 0:
            copy_ms = job.copy_start_event.elapsed_time(job.copy_end_event)
        else:
            copy_ms = 0.0

        if job.use_predicted_topk and job.lookahead_topk is not None:
            bind_topk = job.lookahead_topk
        else:
            bind_topk = std_topk

        remapped_ids = remap_topk_ids(bind_topk.topk_ids, job.active_ids)
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

        num_staged = int(job.active_ids.numel())
        logger.info(
            "[expert_vm] Expert fetch layer=%d active_experts=%d "
            "copy=%.2f ms wait=%.2f ms bytes=%.2f MiB",
            layer_id,
            num_staged,
            copy_ms,
            wait_ms,
            job.bytes_transferred / (1024**2),
        )
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

        # logger.info(
        #     "[expert_vm] Released GPU expert staging for layer=%d",
        #     layer_id,
        # )
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


def expert_vm_wait_and_bind(
    layer: FusedMoE, topk_output: TopKOutput
) -> TopKOutput:
    return get_expert_vm_coordinator().wait_and_bind(layer, topk_output)


def expert_vm_release(layer: FusedMoE) -> None:
    get_expert_vm_coordinator().release(layer)
