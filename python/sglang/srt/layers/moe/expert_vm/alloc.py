from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.layers.moe.expert_vm.config import get_expert_vm_config
from sglang.srt.utils import is_pin_memory_available

logger = logging.getLogger(__name__)

_logged_layers: set[int] = set()


def _layer_id(layer: nn.Module) -> Optional[int]:
    if hasattr(layer, "layer_id"):
        return int(layer.layer_id)
    moe_cfg = getattr(layer, "moe_runner_config", None)
    if moe_cfg is not None:
        return int(moe_cfg.layer_id)
    return None


def should_allocate_expert_weights_on_cpu(layer: nn.Module) -> bool:
    """True when expert VM is on and this MoE layer is not GPU-resident."""
    config = get_expert_vm_config()
    if config is None:
        return False
    layer_id = _layer_id(layer)
    if layer_id is None:
        return False
    return not config.is_resident_layer(layer_id)


def empty_expert_weight(
    layer: nn.Module,
    size: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate an expert weight tensor (GPU by default; CPU+pinned under expert VM)."""
    if should_allocate_expert_weights_on_cpu(layer):
        pin_memory = is_pin_memory_available()
        layer_id = _layer_id(layer)
        if layer_id is not None and layer_id not in _logged_layers:
            _logged_layers.add(layer_id)
            # logger.info(
            #     "[expert_vm] Layer %d expert weights will be allocated in CPU RAM.",
            #     layer_id,
            # )
        return torch.empty(size, dtype=dtype, device="cpu", pin_memory=pin_memory)

    config = get_expert_vm_config()
    layer_id = _layer_id(layer)
    if config is not None and layer_id is not None and layer_id not in _logged_layers:
        _logged_layers.add(layer_id)
        # logger.info(
        #     "[expert_vm] Layer %d expert weights will stay in GPU VRAM (resident).",
        #     layer_id,
        # )
    return torch.empty(size, dtype=dtype)
