"""Router-predicted per-expert prefetch for MoE models (YALIS-style overlap).

Schedule (``num_prefetch_layers=2``):

  1. ``before_experts(L)``: wait/bind L; predict gate+topk for the next
     offloaded layers not already in flight (usually just L+2).
  2. MoE(L) runs alone — no concurrent PCIe.
  3. ``after_experts(L)``: enqueue H2D for those layers.  L+2 hides behind
     attn(L+1)+MoE(L+1)+attn(L+2).  L+1 was already launched from L-1.

Uses ``num_prefetch_layers + 1`` GPU buffer sets (YALIS).  CPU expert storage
is contiguous pinned memory so each row is one linear ``cudaMemcpyAsync``.

Enable via ``--enable-expert-prefetch``.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.utils.layer_timing import step as layer_timing_step

logger = logging.getLogger(__name__)

# cudaMemcpyKind: cudaMemcpyHostToDevice = 1
_CUDA_MEMCPY_H2D = 1
_cudart = None

# How many offloaded layers to prefetch ahead of the current MoE.
_DEFAULT_PREFETCH_LAYERS = 2


def _get_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    try:
        _cudart = ctypes.CDLL("libcudart.so")
        _cudart.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _cudart.cudaMemcpyAsync.restype = ctypes.c_int
    except OSError:
        _cudart = False  # type: ignore[assignment]
    return _cudart


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
    """YALIS-style predicted per-expert prefetch for one MoE model."""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        num_experts: int,
        top_k: int,
        resident_layers: Set[int],
        num_prefetch_layers: int = _DEFAULT_PREFETCH_LAYERS,
        get_gate=lambda layer: layer.mlp.gate,
        get_topk=lambda layer: layer.mlp.topk,
        get_experts=lambda layer: layer.mlp.experts,
        is_sparse=lambda layer: hasattr(getattr(layer, "mlp", None), "experts"),
    ) -> None:
        self.layers = layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.resident_layers = resident_layers
        self.num_prefetch_layers = max(1, int(num_prefetch_layers))
        self._get_gate = get_gate
        self._get_topk = get_topk
        self._get_experts = get_experts
        self._is_sparse = is_sparse

        self._initialized = False
        self._device: Optional[torch.device] = None
        self._transfer_stream: Optional[torch.cuda.Stream] = None

        self._param_names: List[str] = []
        self._cpu_store: Dict[int, Dict[str, torch.Tensor]] = {}
        self._offloaded_layers: List[int] = []
        self._gpu_layout: Dict[str, Tuple[tuple, tuple, torch.dtype]] = {}

        # YALIS: num_buffer_sets = num_prefetch_layers + 1
        self._num_buffers = self.num_prefetch_layers + 1
        self._buffers: List[Dict[str, torch.Tensor]] = []
        self._next_buf = 0

        self._layer_buf: Dict[int, int] = {}
        self._event_pool: Dict[int, torch.cuda.Event] = {}
        self._transfer_events: Dict[int, torch.cuda.Event] = {}
        self._predicted_topk: Dict[int, object] = {}
        # Layers whose H2D has been enqueued (or is about to be).
        self._dma_started: Set[int] = set()

        self._ids_pin: Optional[torch.Tensor] = None
        # Prepared at before_experts; consumed by after_experts.
        self._pending_prefetches: List[dict] = []

        self.stat_layers = 0
        self.stat_predicted = 0
        self.stat_cold = 0
        self._use_batch_copy: Optional[bool] = None

    # ------------------------------------------------------------------ init

    def _resolve_batch_copy(self) -> bool:
        if self._use_batch_copy is not None:
            return self._use_batch_copy
        try:
            from sgl_kernel.expert_prefetch import expert_prefetch_copy  # noqa: F401

            self._use_batch_copy = True
            logger.info("[expert_prefetch] using sgl_kernel batched H2D copy")
        except (ImportError, AttributeError, RuntimeError):
            self._use_batch_copy = False
            logger.info(
                "[expert_prefetch] sgl_kernel batched copy unavailable; "
                "using contiguous pinned cudaMemcpyAsync"
            )
        return self._use_batch_copy

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
        self._transfer_stream = torch.cuda.Stream(device=self._device)

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

        sample_experts = self._get_experts(self.layers[self._offloaded_layers[0]])
        self._param_names = self._detect_param_names(sample_experts)
        assert self._param_names, "could not detect per-expert weight parameters"

        for name in self._param_names:
            p = getattr(sample_experts, name)
            self._gpu_layout[name] = (
                tuple(p.data.size()),
                tuple(p.data.stride()),
                p.data.dtype,
            )

        pin = True
        moved_bytes = 0
        for layer_id in self._offloaded_layers:
            experts = self._get_experts(self.layers[layer_id])
            store: Dict[str, torch.Tensor] = {}
            for name in self._param_names:
                p = getattr(experts, name)
                cpu = self._to_pinned_contiguous(p.data, pin)
                pin = cpu.is_pinned()
                store[name] = cpu
                moved_bytes += cpu.numel() * cpu.element_size()
                p.data = torch.empty(0, dtype=cpu.dtype, device=self._device)
            self._cpu_store[layer_id] = store
            self._event_pool[layer_id] = torch.cuda.Event()

        for _ in range(self._num_buffers):
            buf = {
                name: torch.empty_strided(
                    size=self._gpu_layout[name][0],
                    stride=self._gpu_layout[name][1],
                    dtype=self._gpu_layout[name][2],
                    device=self._device,
                )
                for name in self._param_names
            }
            self._buffers.append(buf)

        self._ids_pin = torch.empty(self.num_experts, dtype=torch.long, pin_memory=True)

        torch.cuda.empty_cache()
        logger.info(
            "[expert_prefetch] init: offloaded %d layers, params=%s, "
            "moved %.2f GB to %s contiguous CPU, %dx GPU buffers (%.2f GB total), "
            "prefetch_ahead=%d, H2D after MoE",
            len(self._offloaded_layers),
            self._param_names,
            moved_bytes / 1024**3,
            "pinned" if pin else "pageable",
            self._num_buffers,
            self._num_buffers * self._buffer_bytes() / 1024**3,
            self.num_prefetch_layers,
        )
        self._initialized = True

    def _buffer_bytes(self) -> int:
        if not self._buffers:
            return 0
        return sum(t.numel() * t.element_size() for t in self._buffers[0].values())

    def _to_pinned_contiguous(self, data: torch.Tensor, pin: bool) -> torch.Tensor:
        host = data.detach().to("cpu", non_blocking=False).contiguous()
        if pin:
            try:
                return host.pin_memory()
            except RuntimeError:
                return host
        return host

    # -------------------------------------------------------------- helpers

    def _expert_ids_cpu(self, topk_ids: torch.Tensor) -> List[int]:
        flat = topk_ids.detach().reshape(-1)
        host = flat.to("cpu", dtype=torch.long, non_blocking=False).tolist()
        return sorted({int(e) for e in host if 0 <= int(e) < self.num_experts})

    def _alloc_buf(self) -> int:
        idx = self._next_buf
        self._next_buf = (self._next_buf + 1) % self._num_buffers
        return idx

    def _next_offloaded_layers(self, layer_id: int, n: int) -> List[int]:
        """Return up to ``n`` offloaded layer ids strictly after ``layer_id``."""
        out: List[int] = []
        nxt = layer_id + 1
        while nxt < len(self.layers) and len(out) < n:
            if nxt in self._cpu_store:
                out.append(nxt)
            nxt += 1
        return out

    def _copy_experts(
        self,
        buf: Dict[str, torch.Tensor],
        store: Dict[str, torch.Tensor],
        ids: List[int],
    ) -> None:
        """Enqueue H2D of selected expert rows on the *current* CUDA stream."""
        if not ids:
            return

        if self._resolve_batch_copy():
            from sgl_kernel.expert_prefetch import expert_prefetch_copy

            n = len(ids)
            id_buf = self._ids_pin[:n]
            for i, e in enumerate(ids):
                id_buf[i] = e
            srcs = [store[name] for name in self._param_names]
            dsts = [buf[name] for name in self._param_names]
            expert_prefetch_copy(srcs, dsts, id_buf, False)
            return

        cudart = _get_cudart()
        if cudart:
            stream = torch.cuda.current_stream().cuda_stream
            for name in self._param_names:
                src = store[name]
                dst = buf[name]
                src_stride = src.stride(0) * src.element_size()
                dst_stride = dst.stride(0) * dst.element_size()
                row_bytes = src[0].numel() * src.element_size()
                src_base = src.data_ptr()
                dst_base = dst.data_ptr()
                for e in ids:
                    err = cudart.cudaMemcpyAsync(
                        ctypes.c_void_p(dst_base + e * dst_stride),
                        ctypes.c_void_p(src_base + e * src_stride),
                        ctypes.c_size_t(row_bytes),
                        _CUDA_MEMCPY_H2D,
                        ctypes.c_void_p(stream),
                    )
                    if err != 0:
                        break
                else:
                    continue
                for e in ids:
                    dst[e].copy_(src[e], non_blocking=True)
            return

        for name in self._param_names:
            src = store[name]
            dst = buf[name]
            for e in ids:
                dst[e].copy_(src[e], non_blocking=True)

    # -------------------------------------------------------------- pipeline

    def before_experts(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        gate: torch.nn.Module,
        topk: torch.nn.Module,
    ):
        """Bind L, predict next ``num_prefetch_layers``.  H2D in ``after_experts``."""
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
            topk_output = self._predicted_topk.pop(layer_id)
            with layer_timing_step("router_bind"):
                self._wait_and_bind(layer_id)
            self.stat_layers += 1
        else:
            with layer_timing_step("router_gate"):
                router_logits, _ = gate(hidden_states)
            with layer_timing_step("router_topk"):
                topk_output = topk(hidden_states, router_logits)
            if layer_id in self._cpu_store:
                self.stat_cold += 1
                with layer_timing_step("router_cold_load"):
                    self._cold_load(layer_id, topk_output.topk_ids)

        # Predict any of the next N offloaded layers not already in flight.
        self._pending_prefetches = []
        for target in self._next_offloaded_layers(
            layer_id, self.num_prefetch_layers
        ):
            if target in self._predicted_topk or target in self._dma_started:
                continue
            item = self._prepare_prefetch(target, hidden_states, from_layer=layer_id)
            if item is not None:
                self._pending_prefetches.append(item)

        return topk_output

    def after_experts(self, layer_id: int) -> None:
        """Enqueue H2D for layers predicted in ``before_experts`` (overlaps attn)."""
        if not self._initialized or not self._offloaded_layers:
            return
        pending = self._pending_prefetches
        if not pending:
            return
        mine = [p for p in pending if p.get("from_layer") == layer_id]
        self._pending_prefetches = [
            p for p in pending if p.get("from_layer") != layer_id
        ]
        if not mine:
            return
        self._start_prefetch_dmas(mine)

    def _wait_and_bind(self, layer_id: int) -> None:
        experts = self._get_experts(self.layers[layer_id])
        buf_idx = self._layer_buf[layer_id]
        event = self._transfer_events.pop(layer_id, None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        self._dma_started.discard(layer_id)
        buf = self._buffers[buf_idx]
        for name in self._param_names:
            getattr(experts, name).data = buf[name]

    def _cold_load(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        buf_idx = self._alloc_buf()
        self._layer_buf[layer_id] = buf_idx
        buf = self._buffers[buf_idx]
        experts = self._get_experts(self.layers[layer_id])
        for name in self._param_names:
            getattr(experts, name).data = buf[name]

        ids = self._expert_ids_cpu(topk_ids)
        store = self._cpu_store[layer_id]
        compute = torch.cuda.current_stream()
        with torch.cuda.stream(self._transfer_stream):
            self._transfer_stream.wait_stream(compute)
            self._copy_experts(buf, store, ids)
        compute.wait_stream(self._transfer_stream)

    def _prepare_prefetch(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        from_layer: int,
    ) -> Optional[dict]:
        if layer_id not in self._cpu_store:
            return None

        gate = self._get_gate(self.layers[layer_id])
        topk_mod = self._get_topk(self.layers[layer_id])
        with torch.no_grad():
            with layer_timing_step("router_predict_gate"):
                router_logits, _ = gate(hidden_states)
            with layer_timing_step("router_predict_topk"):
                topk_output = topk_mod(hidden_states, router_logits)

        predicted_ids = self._expert_ids_cpu(topk_output.topk_ids)
        self.stat_predicted += len(predicted_ids)
        self._predicted_topk[layer_id] = topk_output

        buf_idx = self._alloc_buf()
        self._layer_buf[layer_id] = buf_idx

        return {
            "layer_id": layer_id,
            "from_layer": from_layer,
            "ids": predicted_ids,
            "buf_idx": buf_idx,
        }

    def _start_prefetch_dmas(self, pending_list: List[dict]) -> None:
        """Enqueue H2D on the transfer stream (sync CPU, async GPU)."""
        compute = torch.cuda.current_stream()

        with layer_timing_step("router_predict_launch"):
            with torch.cuda.stream(self._transfer_stream):
                self._transfer_stream.wait_stream(compute)
                for pending in pending_list:
                    layer_id = pending["layer_id"]
                    ids = pending["ids"]
                    buf_idx = pending["buf_idx"]
                    buf = self._buffers[buf_idx]
                    store = self._cpu_store[layer_id]
                    event = self._event_pool[layer_id]
                    self._transfer_events[layer_id] = event
                    self._dma_started.add(layer_id)
                    self._copy_experts(buf, store, ids)
                    event.record(self._transfer_stream)

    def log_stats(self) -> None:
        if self.stat_layers == 0:
            return
        avg = self.stat_predicted / max(1, self.stat_layers)
        logger.info(
            "[expert_prefetch] predicted-routing layers=%d avg_experts_prefetched=%.1f "
            "cold_starts=%d prefetch_ahead=%d",
            self.stat_layers,
            avg,
            self.stat_cold,
            self.num_prefetch_layers,
        )
