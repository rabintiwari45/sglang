"""Optional per-transformer-layer step timing for debugging.

Enable with environment variable ``SGLANG_LOG_LAYER_TIMING=1``.

Each layer logs one line with step times in pipeline order:
  input_norm -> attn -> post_attn_norm -> router -> moe -> post
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch

logger = logging.getLogger(__name__)

_ENABLED: Optional[bool] = None
_ACTIVE: Optional["LayerTimingState"] = None

STEP_ORDER: List[str] = [
    "input_norm",
    "attn",
    "post_attn_norm",
    "router",
    "moe",
    "mlp",
    "post",
]


def is_layer_timing_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        raw = os.environ.get("SGLANG_LOG_LAYER_TIMING", "0").strip().lower()
        _ENABLED = raw in ("1", "true", "yes", "on")
    return _ENABLED


def should_record_layer_timing() -> bool:
    """Skip during torch.compile / piecewise CUDA graph capture."""
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

    def log(self) -> None:
        if not self.steps_ms:
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

    sync_compute_stream()
    start = time.perf_counter()
    try:
        yield
    finally:
        sync_compute_stream()
        _ACTIVE.record(name, (time.perf_counter() - start) * 1000.0)
