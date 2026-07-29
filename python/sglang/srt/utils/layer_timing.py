"""Optional transformer timing for debugging.

Enable with ``SGLANG_LOG_LAYER_TIMING=1``.

Logs one summary line per forward pass (prefill or decode) with aggregated:
  attn, router, moe, moe_compute, total

When router sub-steps are recorded, the summary also breaks ``router`` down into:
  gate, topk, bind, cold_load, predict_gate, predict_topk, predict_launch

Per-layer detail is optional via ``SGLANG_LOG_LAYER_TIMING_DETAIL=1``.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_ENABLED: Optional[bool] = None
_DETAIL: Optional[bool] = None
_ACTIVE: Optional["LayerTimingState"] = None
_FORWARD: Optional["ForwardPassTiming"] = None

STEP_ORDER: List[str] = [
    "input_norm",
    "attn",
    "post_attn_norm",
    "router",
    "router_gate",
    "router_topk",
    "router_bind",
    "router_cold_load",
    "router_predict_gate",
    "router_predict_topk",
    "router_predict_launch",
    "moe",
    "moe_compute",
    "mlp",
    "post",
]

_SUMMARY_STEPS = frozenset({"attn", "router", "moe", "moe_compute"})

# Sub-steps shown in the router breakdown.  Critical ones (before MoE) roll
# into the top-level ``router`` total.
_ROUTER_SUBSTEPS: List[str] = [
    "router_gate",
    "router_topk",
    "router_bind",
    "router_cold_load",
    "router_predict_gate",
    "router_predict_topk",
    "router_predict_launch",
]
# All router substeps that consume wall time on the forward critical path.
# ``predict_launch`` runs after MoE but still blocks the Python thread before
# attn starts, so it must count somewhere other than the residual ``other``.
_ROUTER_CRITICAL_SUBSTEPS = frozenset(
    {
        "router_gate",
        "router_topk",
        "router_bind",
        "router_cold_load",
        "router_predict_gate",
        "router_predict_topk",
        "router_predict_launch",
    }
)
_ROUTER_SUBSTEP_LABELS = {
    "router_gate": "gate",
    "router_topk": "topk",
    "router_bind": "bind",
    "router_cold_load": "cold_load",
    "router_predict_gate": "predict_gate",
    "router_predict_topk": "predict_topk",
    "router_predict_launch": "predict_launch",
}

# ``predict_launch`` only submits a background H2D thread — must not sync
# (would wait on PCIe).  Predict gate/topk run on the compute stream before
# MoE and must sync so their time is not billed to ``moe``.
_NO_SYNC_STEPS = frozenset(
    {
        "router_predict_launch",
    }
)


def is_layer_timing_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        raw = os.environ.get("SGLANG_LOG_LAYER_TIMING", "0").strip().lower()
        _ENABLED = raw in ("1", "true", "yes", "on")
    return _ENABLED


def is_layer_timing_detail_enabled() -> bool:
    global _DETAIL
    if _DETAIL is None:
        raw = os.environ.get("SGLANG_LOG_LAYER_TIMING_DETAIL", "0").strip().lower()
        _DETAIL = raw in ("1", "true", "yes", "on")
    return _DETAIL


def should_record_layer_timing() -> bool:
    if not is_layer_timing_enabled():
        return False
    try:
        from sglang.srt.compilation.piecewise_context_manager import (
            is_in_pcg_torch_compile,
        )

        if is_in_pcg_torch_compile():
            return False
    except ImportError:
        pass
    if torch.compiler.is_compiling():
        return False
    return True


def sync_compute_stream() -> None:
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


@dataclass
class LayerTimingState:
    layer_id: int
    steps_ms: Dict[str, float] = field(default_factory=dict)

    def record(self, name: str, elapsed_ms: float) -> None:
        self.steps_ms[name] = self.steps_ms.get(name, 0.0) + elapsed_ms
        if _FORWARD is None:
            return
        if name in _SUMMARY_STEPS:
            _FORWARD.accumulate(name, elapsed_ms)
        elif name in _ROUTER_SUBSTEPS:
            _FORWARD.accumulate(name, elapsed_ms)
            if name in _ROUTER_CRITICAL_SUBSTEPS:
                _FORWARD.accumulate("router", elapsed_ms)

    def log(self) -> None:
        if not is_layer_timing_detail_enabled() or not self.steps_ms:
            return
        parts = []
        for name in STEP_ORDER:
            if name in self.steps_ms:
                parts.append(f"{name}={self.steps_ms[name]:.2f}")
        for name, ms in self.steps_ms.items():
            if name not in STEP_ORDER:
                parts.append(f"{name}={ms:.2f}")
        total = sum(self.steps_ms.values())
        logger.info(
            "[layer_timing] layer=%d %s total=%.2f ms",
            self.layer_id,
            " ".join(parts),
            total,
        )


@dataclass
class ForwardPassTiming:
    phase: str
    num_tokens: int
    batch_size: int
    steps_ms: Dict[str, float] = field(default_factory=dict)
    _t0: float = 0.0

    def accumulate(self, name: str, elapsed_ms: float) -> None:
        self.steps_ms[name] = self.steps_ms.get(name, 0.0) + elapsed_ms

    def _router_breakdown(self) -> str:
        parts = []
        for sub in _ROUTER_SUBSTEPS:
            ms = self.steps_ms.get(sub)
            if ms is not None and ms > 0.0:
                parts.append(f"{_ROUTER_SUBSTEP_LABELS[sub]}={ms:.2f}")
        if not parts:
            return ""
        return " [" + ", ".join(parts) + "]"

    def log(self) -> None:
        sync_compute_stream()
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        attn = self.steps_ms.get("attn", 0.0)
        router = self.steps_ms.get("router", 0.0)
        moe = self.steps_ms.get("moe", 0.0)
        moe_compute = self.steps_ms.get("moe_compute", 0.0)
        other = max(0.0, total_ms - attn - router - moe)
        router_detail = self._router_breakdown()
        logger.info(
            "[layer_timing] %s ntok=%d bs=%d "
            "attn=%.2f ms router=%.2f ms%s moe=%.2f ms moe_compute=%.2f ms "
            "other=%.2f ms total=%.2f ms",
            self.phase,
            self.num_tokens,
            self.batch_size,
            attn,
            router,
            router_detail,
            moe,
            moe_compute,
            other,
            total_ms,
        )


def _phase_label(forward_batch: "ForwardBatch") -> str:
    mode = forward_batch.forward_mode
    if mode.is_prefill():
        return "prefill"
    if mode.is_decode():
        return "decode"
    return mode.name.lower()


def _num_tokens(forward_batch: "ForwardBatch") -> int:
    if forward_batch.forward_mode.is_prefill():
        ext = forward_batch.extend_num_tokens
        if ext is not None and ext > 0:
            return int(ext)
    return int(forward_batch.batch_size)


def begin_forward(forward_batch: "ForwardBatch") -> None:
    global _FORWARD
    if not should_record_layer_timing():
        return
    sync_compute_stream()
    _FORWARD = ForwardPassTiming(
        phase=_phase_label(forward_batch),
        num_tokens=_num_tokens(forward_batch),
        batch_size=int(forward_batch.batch_size),
        _t0=time.perf_counter(),
    )


def end_forward() -> None:
    global _FORWARD
    if _FORWARD is None:
        return
    _FORWARD.log()
    _FORWARD = None


@contextmanager
def layer(layer_id: int) -> Iterator[None]:
    global _ACTIVE
    if not should_record_layer_timing():
        yield
        return

    state = LayerTimingState(layer_id=layer_id)
    _ACTIVE = state
    try:
        yield
    finally:
        state.log()
        _ACTIVE = None


@contextmanager
def step(name: str) -> Iterator[None]:
    if _ACTIVE is None or not should_record_layer_timing():
        yield
        return

    no_sync = name in _NO_SYNC_STEPS
    if not no_sync:
        sync_compute_stream()
    start = time.perf_counter()
    try:
        yield
    finally:
        if not no_sync:
            sync_compute_stream()
        _ACTIVE.record(name, (time.perf_counter() - start) * 1000.0)
