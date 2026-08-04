"""Router-predicted per-expert prefetch for MoE models (YALIS-style overlap).

Pipeline (``num_prefetch_layers=2``), all on the forward (main) thread -- a
Python worker thread was tried and rejected: its GIL usage stalls the
launch-bound decode loop and inflates every layer.

For each MoE layer L:

  1. ``before_experts(L)``: wait/bind L's expert buffer (GPU-side event wait);
     enqueue gate+topk prediction kernels for the next offloaded layers
     (usually just L+2) plus an async D2H copy of the predicted expert ids
     into a pinned ring slot (CUDA event recorded, no host sync).
  2. MoE(L) runs.
  3. ``after_experts(L)``: record "L's buffer free" event; then process the
     prediction made in step 1: the id-copy event has already fired (the MoE
     timing sync passed it), so reading the pinned ids costs nothing.  Expert
     rows already in the per-layer GPU cache are restored with device-to-device
     memcpys; missing rows are fetched from contiguous pinned CPU memory with
     one batched H2D call and inserted into the LRU cache.  All copies run on
     the transfer stream guarded by per-buffer CUDA events -- no stream-wide
     synchronization -- so the H2D overlaps MoE(L)..attn(L+2) compute.

The per-layer GPU expert cache is the key addition over pure prefetch: PCIe
H2D bandwidth (~8 GB/s measured on this host) makes a full 8-expert layer
transfer (~18 MB) cost ~2.3 ms, more than a layer's compute (~1.1 ms), so
overlap alone cannot hide it.  Cache hits skip PCIe entirely.

Uses ``num_prefetch_layers + 1`` GPU buffer sets (YALIS) with a static
layer->buffer mapping.

Enable via ``--enable-expert-prefetch``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from sglang.srt.utils.layer_timing import step as layer_timing_step

logger = logging.getLogger(__name__)

# How many offloaded layers to prefetch ahead of the current MoE.
_DEFAULT_PREFETCH_LAYERS = 2
# Default per-layer GPU expert-cache capacity (rows). 0 disables the cache.
_DEFAULT_CACHE_SLOTS = 32
# Pinned ring used to move predicted topk ids GPU->CPU asynchronously.
_IDS_RING_SLOTS = 8
_IDS_SLOT_CAP = 4096  # max (tokens * top_k) handled asynchronously

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


class _LayerCache:
    """Per-layer GPU cache of expert rows with LRU eviction."""

    __slots__ = ("tensors", "id2slot", "lru", "free")

    def __init__(self, tensors: Dict[str, torch.Tensor], capacity: int):
        self.tensors = tensors
        self.id2slot: Dict[int, int] = {}
        # expert_id -> None; least recently used first.
        self.lru: "OrderedDict[int, None]" = OrderedDict()
        self.free: List[int] = list(range(capacity - 1, -1, -1))


class ExpertPrefetcher:
    """YALIS-style predicted per-expert prefetch with a GPU expert cache."""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        num_experts: int,
        top_k: int,
        resident_layers: Set[int],
        num_prefetch_layers: int = _DEFAULT_PREFETCH_LAYERS,
        cache_slots: int = _DEFAULT_CACHE_SLOTS,
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
        self.cache_slots = max(0, min(int(cache_slots), num_experts))
        self._get_gate = get_gate
        self._get_topk = get_topk
        self._get_experts = get_experts
        self._is_sparse = is_sparse

        self._initialized = False
        self._device: Optional[torch.device] = None
        self._transfer_stream: Optional[torch.cuda.Stream] = None

        self._param_names: List[str] = []
        self._cpu_store: Dict[int, Dict[str, torch.Tensor]] = {}
        # Per-layer lists in self._param_names order (avoids per-layer dict
        # walks on the hot path).
        self._cpu_lists: Dict[int, List[torch.Tensor]] = {}
        self._offloaded_layers: List[int] = []
        self._gpu_layout: Dict[str, Tuple[tuple, tuple, torch.dtype]] = {}

        # YALIS: num_buffer_sets = num_prefetch_layers + 1
        self._num_buffers = self.num_prefetch_layers + 1
        self._buffers: List[Dict[str, torch.Tensor]] = []
        self._buffer_lists: List[List[torch.Tensor]] = []
        # Static layer -> buffer-set mapping.
        self._layer_buf: Dict[int, int] = {}
        # Per-buffer "last MoE using this buffer was enqueued" events.
        self._buf_ready: List[torch.cuda.Event] = []

        # Per-layer transfer-complete CUDA events.
        self._xfer_events: Dict[int, torch.cuda.Event] = {}
        self._predicted_topk: Dict[int, object] = {}
        # Predictions awaiting DMA enqueue: list of dicts.
        self._pending: List[dict] = []

        # Per-layer GPU expert cache.
        self._cache: Dict[int, _LayerCache] = {}
        self._cache_lists: Dict[int, List[torch.Tensor]] = {}

        # Pinned ring for async id copies.
        self._ids_ring: Optional[torch.Tensor] = None
        self._ids_ring_np = None
        self._ids_ring_events: List[torch.cuda.Event] = []
        self._ids_ring_free: deque = deque()

        # Shared pinned index staging (values are consumed on the host at
        # enqueue time by the copy ops, so one set is enough).
        self._stage: Dict[str, torch.Tensor] = {}
        self._stage_np: Dict[str, np.ndarray] = {}

        self.stat_layers = 0
        self.stat_hits = 0
        self.stat_misses = 0
        self.stat_cold = 0
        self.stat_transfers = 0
        self._use_batch_copy: Optional[bool] = None

    # ------------------------------------------------------------------ init

    def _resolve_batch_copy(self) -> bool:
        if self._use_batch_copy is not None:
            return self._use_batch_copy
        try:
            from sgl_kernel.expert_prefetch import (  # noqa: F401
                expert_cache_copy,
                expert_prefetch_copy,
            )

            self._use_batch_copy = True
            logger.info("[expert_prefetch] using sgl_kernel batched copies")
        except (ImportError, AttributeError, RuntimeError):
            self._use_batch_copy = False
            logger.info(
                "[expert_prefetch] sgl_kernel batched copy unavailable; "
                "using per-row copies"
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
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        if self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
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
            self._cpu_lists[layer_id] = [store[n] for n in self._param_names]
            self._xfer_events[layer_id] = torch.cuda.Event()

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
            self._buffer_lists.append([buf[n] for n in self._param_names])

        # Static layer -> buffer mapping (reuse distance == num_buffers).
        for i, layer_id in enumerate(self._offloaded_layers):
            self._layer_buf[layer_id] = i % self._num_buffers

        # Pre-record buffer-free events so the first waits are no-ops.
        self._buf_ready = [torch.cuda.Event() for _ in range(self._num_buffers)]
        for ev in self._buf_ready:
            ev.record()

        # Per-layer GPU expert cache.
        cache_bytes = 0
        if self.cache_slots > 0:
            for layer_id in self._offloaded_layers:
                tensors = {}
                for name in self._param_names:
                    shape = (self.cache_slots,) + tuple(
                        self._gpu_layout[name][0][1:]
                    )
                    t = torch.empty(
                        shape,
                        dtype=self._gpu_layout[name][2],
                        device=self._device,
                    )
                    cache_bytes += t.numel() * t.element_size()
                    tensors[name] = t
                self._cache[layer_id] = _LayerCache(tensors, self.cache_slots)
                self._cache_lists[layer_id] = [
                    tensors[n] for n in self._param_names
                ]

        # Pinned ring for async id copies.
        self._ids_ring = torch.empty(
            (_IDS_RING_SLOTS, _IDS_SLOT_CAP), dtype=torch.long, pin_memory=True
        )
        self._ids_ring_np = self._ids_ring.numpy()
        self._ids_ring_events = [
            torch.cuda.Event() for _ in range(_IDS_RING_SLOTS)
        ]
        self._ids_ring_free = deque(range(_IDS_RING_SLOTS))

        # Shared pinned index staging.  The copy ops read the row indices on
        # the host at enqueue time, so the staging can be reused immediately
        # after each call returns.
        n = self.num_experts
        for key in ("miss", "hit_ids", "hit_slots", "ins_ids", "ins_slots"):
            t = torch.empty(n, dtype=torch.long, pin_memory=True)
            self._stage[key] = t
            self._stage_np[key] = t.numpy()

        torch.cuda.empty_cache()
        logger.info(
            "[expert_prefetch] init: offloaded %d layers, params=%s, "
            "moved %.2f GB to %s contiguous CPU, %dx GPU buffers (%.2f GB), "
            "cache %d slots/layer (%.2f GB), prefetch_ahead=%d, inline enqueue",
            len(self._offloaded_layers),
            self._param_names,
            moved_bytes / 1024**3,
            "pinned" if pin else "pageable",
            self._num_buffers,
            self._num_buffers * self._buffer_bytes() / 1024**3,
            self.cache_slots,
            cache_bytes / 1024**3,
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
        host = flat.to("cpu", dtype=torch.long, non_blocking=False).numpy()
        ids = np.unique(host)
        return [int(e) for e in ids if 0 <= e < self.num_experts]

    def _next_offloaded_layers(self, layer_id: int, n: int) -> List[int]:
        """Return up to ``n`` offloaded layer ids strictly after ``layer_id``."""
        out: List[int] = []
        nxt = layer_id + 1
        while nxt < len(self.layers) and len(out) < n:
            if nxt in self._cpu_store:
                out.append(nxt)
            nxt += 1
        return out

    # -------------------------------------------------------------- pipeline

    def before_experts(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        gate: torch.nn.Module,
        topk: torch.nn.Module,
    ):
        """Bind L, enqueue predictions for the next offloaded layers."""
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
                    ids = self._expert_ids_cpu(topk_output.topk_ids)
                    self._enqueue_transfer(layer_id, ids)
                    self._wait_and_bind(layer_id)

        # Predict any of the next N offloaded layers not already in flight.
        for target in self._next_offloaded_layers(
            layer_id, self.num_prefetch_layers
        ):
            if target in self._predicted_topk:
                continue
            self._predict(target, hidden_states, from_layer=layer_id)

        return topk_output

    def after_experts(self, layer_id: int) -> None:
        """Mark L's buffer reusable and enqueue DMAs for L's predictions."""
        if not self._initialized or not self._offloaded_layers:
            return
        buf_idx = self._layer_buf.get(layer_id)
        if buf_idx is not None:
            self._buf_ready[buf_idx].record()

        if not self._pending:
            return
        mine = [p for p in self._pending if p["from_layer"] == layer_id]
        if not mine:
            return
        self._pending = [p for p in self._pending if p["from_layer"] != layer_id]
        with layer_timing_step("router_predict_launch"):
            for item in mine:
                ids = item["ids"]
                if ids is None:
                    # The id-copy event has fired by now (the MoE timing sync
                    # passed the prediction kernels), so this wait is ~free.
                    item["event"].synchronize()
                    slot = item["slot"]
                    raw = self._ids_ring_np[slot, : item["n_ids"]]
                    ids_arr = np.unique(raw)
                    self._ids_ring_free.append(slot)
                    ids = [
                        int(e) for e in ids_arr if 0 <= e < self.num_experts
                    ]
                self._enqueue_transfer(item["layer_id"], ids)

    def _predict(
        self, layer_id: int, hidden_states: torch.Tensor, from_layer: int
    ) -> None:
        gate = self._get_gate(self.layers[layer_id])
        topk_mod = self._get_topk(self.layers[layer_id])
        with torch.no_grad():
            with layer_timing_step("router_predict_gate"):
                router_logits, _ = gate(hidden_states)
            with layer_timing_step("router_predict_topk"):
                topk_output = topk_mod(hidden_states, router_logits)

        self._predicted_topk[layer_id] = topk_output

        flat = topk_output.topk_ids.detach().reshape(-1)
        n = flat.numel()
        if n <= _IDS_SLOT_CAP and self._ids_ring_free:
            slot = self._ids_ring_free.popleft()
            # Async D2H into pinned ring; consumed in after_experts.
            self._ids_ring[slot, :n].copy_(flat.to(torch.long), non_blocking=True)
            ev = self._ids_ring_events[slot]
            ev.record()
            self._pending.append(
                {
                    "layer_id": layer_id,
                    "from_layer": from_layer,
                    "ids": None,
                    "slot": slot,
                    "n_ids": n,
                    "event": ev,
                }
            )
        else:
            # Large batch (prefill) or ring exhausted: sync extraction.
            ids = self._expert_ids_cpu(topk_output.topk_ids)
            self._pending.append(
                {
                    "layer_id": layer_id,
                    "from_layer": from_layer,
                    "ids": ids,
                }
            )

    def _wait_and_bind(self, layer_id: int) -> None:
        torch.cuda.current_stream().wait_event(self._xfer_events[layer_id])
        experts = self._get_experts(self.layers[layer_id])
        buf = self._buffers[self._layer_buf[layer_id]]
        for name in self._param_names:
            getattr(experts, name).data = buf[name]

    # ------------------------------------------------------------- transfers

    def _enqueue_transfer(self, layer_id: int, ids: List[int]) -> None:
        """Enqueue cache restores + H2D for ``ids`` on the transfer stream."""
        cache = self._cache.get(layer_id)
        hits: List[int] = []
        misses: List[int] = []
        if cache is not None:
            id2slot = cache.id2slot
            for e in ids:
                if e in id2slot:
                    hits.append(e)
                else:
                    misses.append(e)
            lru = cache.lru
            for e in hits:
                lru.move_to_end(e)
        else:
            misses = list(ids)

        buf_idx = self._layer_buf[layer_id]
        buf_list = self._buffer_lists[buf_idx]
        use_batch = self._resolve_batch_copy()

        with torch.cuda.stream(self._transfer_stream):
            # Do not overwrite the buffer before its previous MoE finished.
            self._transfer_stream.wait_event(self._buf_ready[buf_idx])

            if hits:
                nh = len(hits)
                self._stage_np["hit_ids"][:nh] = hits
                self._stage_np["hit_slots"][:nh] = [
                    cache.id2slot[e] for e in hits
                ]
                self._d2d_rows(
                    self._cache_lists[layer_id],
                    buf_list,
                    self._stage["hit_slots"][:nh],
                    self._stage["hit_ids"][:nh],
                    use_batch,
                )

            if misses:
                nm = len(misses)
                self._stage_np["miss"][:nm] = misses
                self._h2d_rows(
                    self._cpu_lists[layer_id],
                    buf_list,
                    self._stage["miss"][:nm],
                    misses,
                    use_batch,
                )

                if cache is not None:
                    victims = self._alloc_cache_slots(cache, misses, ids)
                    nv = len(victims)
                    if nv:
                        ins = misses[:nv]
                        self._stage_np["ins_ids"][:nv] = ins
                        self._stage_np["ins_slots"][:nv] = victims
                        self._d2d_rows(
                            buf_list,
                            self._cache_lists[layer_id],
                            self._stage["ins_ids"][:nv],
                            self._stage["ins_slots"][:nv],
                            use_batch,
                        )
                        id2slot = cache.id2slot
                        lru = cache.lru
                        for e, s in zip(ins, victims):
                            id2slot[e] = s
                            lru[e] = None

            self._xfer_events[layer_id].record(self._transfer_stream)

        self.stat_hits += len(hits)
        self.stat_misses += len(misses)
        self.stat_transfers += 1
        if self.stat_transfers % 470 == 0:  # every ~10 decode tokens
            self.log_stats()

    def _h2d_rows(
        self,
        srcs: List[torch.Tensor],
        dsts: List[torch.Tensor],
        ids_pin: torch.Tensor,
        ids: List[int],
        use_batch: bool,
    ) -> None:
        if use_batch:
            from sgl_kernel.expert_prefetch import expert_prefetch_copy

            expert_prefetch_copy(srcs, dsts, ids_pin, False)
            return
        for src, dst in zip(srcs, dsts):
            for e in ids:
                dst[e].copy_(src[e], non_blocking=True)

    def _d2d_rows(
        self,
        srcs: List[torch.Tensor],
        dsts: List[torch.Tensor],
        src_rows: torch.Tensor,
        dst_rows: torch.Tensor,
        use_batch: bool,
    ) -> None:
        if use_batch:
            from sgl_kernel.expert_prefetch import expert_cache_copy

            expert_cache_copy(srcs, dsts, src_rows, dst_rows)
            return
        s_rows = src_rows.tolist()
        d_rows = dst_rows.tolist()
        for src, dst in zip(srcs, dsts):
            for s, d in zip(s_rows, d_rows):
                dst[d].copy_(src[s], non_blocking=True)

    def _alloc_cache_slots(
        self, cache: _LayerCache, misses: List[int], current_ids: List[int]
    ) -> List[int]:
        """Pick cache slots for miss inserts: free slots, then LRU eviction."""
        victims: List[int] = []
        cur = set(current_ids)
        for _ in range(len(misses)):
            if cache.free:
                victims.append(cache.free.pop())
                continue
            evicted = None
            for e in cache.lru:  # oldest first
                if e not in cur:
                    evicted = e
                    break
            if evicted is None:
                break
            victims.append(cache.id2slot.pop(evicted))
            del cache.lru[evicted]
        return victims

    # ------------------------------------------------------------------ misc

    def log_stats(self) -> None:
        total = self.stat_hits + self.stat_misses
        if total == 0:
            return
        logger.info(
            "[expert_prefetch] layers=%d cache_hits=%d misses=%d "
            "hit_rate=%.1f%% cold_starts=%d prefetch_ahead=%d",
            self.stat_layers,
            self.stat_hits,
            self.stat_misses,
            100.0 * self.stat_hits / total,
            self.stat_cold,
            self.num_prefetch_layers,
        )
