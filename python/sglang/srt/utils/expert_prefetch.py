"""Router-predicted per-expert prefetch for MoE models (predicted-routing mode).

This implements a layer-wise prefetching pipeline that lets a Mixture-of-Experts
model run with its expert weights stored in CPU RAM while only a small,
double-buffered working set lives in GPU VRAM:

  * All expert weights are moved to pinned CPU memory (except a small set of
    "resident" layers, e.g. layer 0, which stay in VRAM).
  * Two GPU "expert buffers" hold one transformer layer's worth of expert
    weights each (double buffering).
  * While layer ``N`` runs its expert computation, layer ``N+1``'s routing is
    *predicted* by running layer ``N+1``'s router (gate + topk) on layer ``N``'s
    hidden states, and exactly the predicted experts are copied from CPU to GPU
    on a side CUDA stream, hiding the PCIe transfer behind compute.
  * **No fallback / predicted-routing only**: layer ``N+1`` then computes using
    that predicted routing directly (it does *not* re-run its own router). Since
    the routing is exactly the set of experts we prefetched, every routed expert
    is guaranteed resident -- there are no misses by construction. This assumes
    the prediction is correct (the true input to layer ``N+1``'s router is only
    available after layer ``N`` finishes), so outputs are an approximation of the
    unmodified model.

The first/resident layer has no predecessor and keeps all its experts in VRAM,
so it uses its own real router.

The Marlin GPTQ MoE kernel (and the fp16/bf16 fused MoE kernel) index expert
weights by *global* expert id, so each GPU buffer keeps the full
``[num_experts, ...]`` shape and we only fill in the rows for the predicted
experts; un-needed rows are left stale and never read.

Enable via ``--enable-expert-prefetch`` (requires ``--disable-cuda-graph`` and
``--disable-piecewise-cuda-graph``).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import torch

from sglang.srt.utils.layer_timing import step as layer_timing_step

logger = logging.getLogger(__name__)

# Per-expert parameters are auto-detected (first dim == num_experts and
# non-empty), so this works for both GPTQ-Marlin (w13_qweight/w2_qweight/
# w13_scales/w2_scales/...) and the plain fused MoE (w13_weight/w2_weight).

_instance: Optional["ExpertPrefetcher"] = None


def get_expert_prefetcher() -> Optional["ExpertPrefetcher"]:
    return _instance


def set_expert_prefetcher(instance: Optional["ExpertPrefetcher"]) -> None:
    global _instance
    _instance = instance


def _parse_resident_layers(spec: str) -> Set[int]:
    out: Set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


class ExpertPrefetcher:
    """Drives the predicted per-expert prefetch pipeline for one model.

    The model is expected to call :meth:`on_moe` from each sparse MoE block,
    right after the real router has produced ``topk_ids`` and immediately before
    the expert computation runs.
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        num_experts: int,
        top_k: int,
        resident_layers: Set[int],
        get_gate=lambda layer: layer.mlp.gate,
        get_topk=lambda layer: layer.mlp.topk,
        get_experts=lambda layer: layer.mlp.experts,
        is_sparse=lambda layer: hasattr(getattr(layer, "mlp", None), "experts"),
    ) -> None:
        self.layers = layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.resident_layers = resident_layers
        self._get_gate = get_gate
        self._get_topk = get_topk
        self._get_experts = get_experts
        self._is_sparse = is_sparse

        self._initialized = False
        self._device: Optional[torch.device] = None
        self._side_stream: Optional[torch.cuda.Stream] = None

        # Per-expert parameter names that we shuttle between CPU and GPU.
        self._param_names: List[str] = []

        # cpu_store[layer_id][param_name] -> pinned CPU tensor of shape [E, ...]
        self._cpu_store: Dict[int, Dict[str, torch.Tensor]] = {}
        # Sparse layer ids that are offloaded (i.e. not resident).
        self._offloaded_layers: List[int] = []

        # Double buffer: list of two dicts {param_name -> cuda tensor [E, ...]}.
        self._buffers: List[Dict[str, torch.Tensor]] = []
        self._toggle = 0

        # Bookkeeping for the in-flight prefetch of each layer.
        self._layer_buf: Dict[int, int] = {}
        self._events: Dict[int, torch.cuda.Event] = {}
        # The routing (TopK output) predicted for each layer one step ahead.
        self._predicted_topk: Dict[int, object] = {}

        # Stats (host-side counters; cheap).
        self.stat_layers = 0
        self.stat_predicted = 0
        self.stat_cold = 0

    # ------------------------------------------------------------------ init

    def _detect_param_names(self, experts: torch.nn.Module) -> List[str]:
        names = []
        for name, p in experts.named_parameters(recurse=False):
            if p is None:
                continue
            if p.dim() >= 1 and p.shape[0] == self.num_experts and p.numel() > 0:
                names.append(name)
        return names

    def maybe_init(self, device) -> None:
        if self._initialized:
            return
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._side_stream = torch.cuda.Stream(device=self._device)

        # Find sparse layers and split into resident vs offloaded.
        sparse_layer_ids = [
            i for i, layer in enumerate(self.layers) if self._is_sparse(layer)
        ]
        self._offloaded_layers = [
            i for i in sparse_layer_ids if i not in self.resident_layers
        ]
        if not self._offloaded_layers:
            logger.warning(
                "[expert_prefetch] no offloaded layers; prefetch is a no-op"
            )
            self._initialized = True
            return

        # Auto-detect the per-expert params from the first offloaded layer.
        sample_experts = self._get_experts(self.layers[self._offloaded_layers[0]])
        self._param_names = self._detect_param_names(sample_experts)
        assert self._param_names, "could not detect per-expert weight parameters"

        pin = True
        moved_bytes = 0
        for layer_id in self._offloaded_layers:
            experts = self._get_experts(self.layers[layer_id])
            store: Dict[str, torch.Tensor] = {}
            for name in self._param_names:
                p = getattr(experts, name)
                cpu = self._to_pinned_cpu(p.data, pin)
                pin = cpu.is_pinned()
                store[name] = cpu
                moved_bytes += cpu.numel() * cpu.element_size()
                # Free the GPU copy; the param will be rebound to a shared
                # buffer at compute time.
                p.data = torch.empty(0, dtype=cpu.dtype, device=self._device)
            self._cpu_store[layer_id] = store

        # Allocate the two GPU buffers, sized to one layer's expert weights.
        sample_store = self._cpu_store[self._offloaded_layers[0]]
        for _ in range(2):
            buf = {
                name: torch.empty_strided(
                    size=t.size(),
                    stride=t.stride(),
                    dtype=t.dtype,
                    device=self._device,
                )
                for name, t in sample_store.items()
            }
            self._buffers.append(buf)

        torch.cuda.empty_cache()
        logger.info(
            "[expert_prefetch] init: offloaded %d layers, params=%s, "
            "moved %.2f GB to %s CPU, 2x GPU buffers (%.2f GB total)",
            len(self._offloaded_layers),
            self._param_names,
            moved_bytes / 1024**3,
            "pinned" if pin else "pageable",
            2 * self._buffer_bytes() / 1024**3,
        )
        self._initialized = True

    def _buffer_bytes(self) -> int:
        if not self._buffers:
            return 0
        return sum(t.numel() * t.element_size() for t in self._buffers[0].values())

    def _to_pinned_cpu(self, data: torch.Tensor, pin: bool) -> torch.Tensor:
        try:
            cpu = torch.empty_strided(
                size=data.size(),
                stride=data.stride(),
                dtype=data.dtype,
                device="cpu",
                pin_memory=pin,
            )
        except RuntimeError:
            cpu = torch.empty_strided(
                size=data.size(),
                stride=data.stride(),
                dtype=data.dtype,
                device="cpu",
            )
        cpu.copy_(data)
        return cpu

    # -------------------------------------------------------------- pipeline

    def before_experts(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        gate: torch.nn.Module,
        topk: torch.nn.Module,
    ):
        """Return the TopK routing to use for this layer's expert computation.

        For a resident/first layer (or a cold start) this runs the layer's own
        real router. For an offloaded layer it returns the routing that was
        *predicted* during the previous layer (and whose experts have been
        prefetched into a GPU buffer), binding the experts module to that buffer.

        It also launches the predicted prefetch for the next layer, which runs on
        a side stream and overlaps with this layer's expert compute.
        """
        # Empty batches (e.g. padding-only): fall back to the plain router path.
        if hidden_states is None or hidden_states.shape[0] == 0:
            with layer_timing_step("router_gate"):
                router_logits, _ = gate(hidden_states)
            with layer_timing_step("router_topk"):
                return topk(hidden_states, router_logits)

        self.maybe_init(hidden_states.device)

        if not self._offloaded_layers:
            with layer_timing_step("router_gate"):
                router_logits, _ = gate(hidden_states)
            with layer_timing_step("router_topk"):
                return topk(hidden_states, router_logits)

        if layer_id in self._predicted_topk:
            # Predicted-routing path: use the routing decided one layer earlier
            # and bind the experts to the buffer that holds those experts.
            topk_output = self._predicted_topk.pop(layer_id)
            with layer_timing_step("router_bind"):
                self._bind_resident(layer_id)
            self.stat_layers += 1
        else:
            # Resident/first layer or cold start: use the real router.
            with layer_timing_step("router_gate"):
                router_logits, _ = gate(hidden_states)
            with layer_timing_step("router_topk"):
                topk_output = topk(hidden_states, router_logits)
            if layer_id in self._cpu_store:
                # Cold start safety (e.g. a non-resident first layer): the
                # experts were never prefetched, so load the routed ones now.
                self.stat_cold += 1
                with layer_timing_step("router_cold_load"):
                    self._cold_load(layer_id, topk_output.topk_ids)

        # Kick off the predicted prefetch for the next sparse layer; this runs
        # on the side stream and overlaps with this layer's expert compute.
        self._launch_prefetch(self._next_offloaded_layer(layer_id), hidden_states)
        return topk_output

    def _next_offloaded_layer(self, layer_id: int) -> Optional[int]:
        nxt = layer_id + 1
        while nxt < len(self.layers):
            if nxt in self._cpu_store:
                return nxt
            nxt += 1
        return None

    def _bind_resident(self, layer_id: int) -> None:
        """Wait for this layer's prefetch and point its experts at the buffer."""
        experts = self._get_experts(self.layers[layer_id])
        buf_idx = self._layer_buf[layer_id]
        event = self._events.get(layer_id)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        buf = self._buffers[buf_idx]
        for name in self._param_names:
            getattr(experts, name).data = buf[name]

    def _cold_load(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        buf_idx = self._toggle
        self._toggle ^= 1
        self._layer_buf[layer_id] = buf_idx
        buf = self._buffers[buf_idx]
        experts = self._get_experts(self.layers[layer_id])
        for name in self._param_names:
            getattr(experts, name).data = buf[name]
        ids = [e for e in torch.unique(topk_ids).tolist() if 0 <= e < self.num_experts]
        store = self._cpu_store[layer_id]
        for name in self._param_names:
            dst = buf[name]
            src = store[name]
            for e in ids:
                dst[e].copy_(src[e], non_blocking=True)

    def _launch_prefetch(
        self, layer_id: Optional[int], hidden_states: torch.Tensor
    ) -> None:
        if layer_id is None or layer_id not in self._cpu_store:
            return

        # Predict the next layer's routing by running its real router (gate +
        # topk) on the current hidden states. This same routing will be used for
        # that layer's expert computation (predicted-routing, no fallback).
        gate = self._get_gate(self.layers[layer_id])
        topk = self._get_topk(self.layers[layer_id])
        with torch.no_grad():
            with layer_timing_step("router_predict_gate"):
                router_logits, _ = gate(hidden_states)
            with layer_timing_step("router_predict_topk"):
                topk_output = topk(hidden_states, router_logits)
        predicted_ids = [
            e
            for e in torch.unique(topk_output.topk_ids).tolist()
            if 0 <= e < self.num_experts
        ]
        self.stat_predicted += len(predicted_ids)
        self._predicted_topk[layer_id] = topk_output

        buf_idx = self._toggle
        self._toggle ^= 1
        self._layer_buf[layer_id] = buf_idx
        buf = self._buffers[buf_idx]
        store = self._cpu_store[layer_id]

        # Make the side stream wait for all prior default-stream work so we
        # never overwrite a buffer that the previous layer is still reading.
        with layer_timing_step("router_predict_launch"):
            self._side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._side_stream):
                for name in self._param_names:
                    dst = buf[name]
                    src = store[name]
                    for e in predicted_ids:
                        dst[e].copy_(src[e], non_blocking=True)
                event = torch.cuda.Event()
                event.record(self._side_stream)
        self._events[layer_id] = event

    def log_stats(self) -> None:
        if self.stat_layers == 0:
            return
        avg = self.stat_predicted / max(1, self.stat_layers)
        logger.info(
            "[expert_prefetch] predicted-routing layers=%d avg_experts_prefetched=%.1f "
            "cold_starts=%d",
            self.stat_layers,
            avg,
            self.stat_cold,
        )
