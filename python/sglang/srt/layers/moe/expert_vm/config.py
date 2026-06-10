from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

import torch

_cached_config: Optional["ExpertVMConfig"] = None


def sync_compute_stream() -> None:
    """Synchronize only the current compute stream, not expert-vm H2D copies."""
    torch.cuda.current_stream().synchronize()


def parse_resident_layer_ids(spec: str) -> Set[int]:
    """Parse comma-separated MoE layer ids that keep experts on GPU.

    Use ``none`` (or ``-``) so every layer's experts stay in CPU RAM only.
    """
    raw = (spec if spec is not None else "0").strip().lower()
    if raw in ("", "none", "null", "-"):
        return set()
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


@dataclass(frozen=True)
class ExpertVMConfig:
    enabled: bool
    resident_layer_ids: Set[int]

    def is_resident_layer(self, layer_id: int) -> bool:
        return layer_id in self.resident_layer_ids


def set_expert_vm_config_from_server_args(server_args) -> None:
    """Cache expert VM settings when scheduler server args are installed."""
    global _cached_config
    if not getattr(server_args, "enable_expert_vm", False):
        _cached_config = None
        return
    _cached_config = ExpertVMConfig(
        enabled=True,
        resident_layer_ids=parse_resident_layer_ids(
            server_args.expert_vm_resident_layers
        ),
    )


def get_expert_vm_config() -> Optional[ExpertVMConfig]:
    if _cached_config is not None:
        return _cached_config
    try:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
    except Exception:
        return None
    if server_args is None or not server_args.enable_expert_vm:
        return None
    return ExpertVMConfig(
        enabled=True,
        resident_layer_ids=parse_resident_layer_ids(
            server_args.expert_vm_resident_layers
        ),
    )


def is_expert_vm_enabled() -> bool:
    cfg = get_expert_vm_config()
    return cfg is not None and cfg.enabled
