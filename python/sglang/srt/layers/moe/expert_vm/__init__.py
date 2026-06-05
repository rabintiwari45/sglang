# Expert virtual memory: CPU-backed MoE experts with layer-wise GPU staging.
# Manager symbols are lazy-loaded to avoid import cycles with FusedMoE / model_config.

from __future__ import annotations

from typing import Any

from sglang.srt.layers.moe.expert_vm.alloc import (
    empty_expert_weight,
    should_allocate_expert_weights_on_cpu,
)
from sglang.srt.layers.moe.expert_vm.config import (
    ExpertVMConfig,
    get_expert_vm_config,
    is_expert_vm_enabled,
    parse_resident_layer_ids,
    set_expert_vm_config_from_server_args,
)

_LAZY_MANAGER_EXPORTS = {
    "ExpertVMManager": "ExpertVMManager",
    "finalize_expert_vm_after_load": "finalize_expert_vm_after_load",
    "get_expert_vm_manager": "get_expert_vm_manager",
    "maybe_release_expert_vm_gpu_weights": "maybe_release_expert_vm_gpu_weights",
    "maybe_stage_expert_vm_gpu_weights": "maybe_stage_expert_vm_gpu_weights",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MANAGER_EXPORTS:
        from sglang.srt.layers.moe.expert_vm import manager as _manager_mod

        return getattr(_manager_mod, _LAZY_MANAGER_EXPORTS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "empty_expert_weight",
    "ExpertVMConfig",
    "ExpertVMManager",
    "finalize_expert_vm_after_load",
    "should_allocate_expert_weights_on_cpu",
    "get_expert_vm_config",
    "get_expert_vm_manager",
    "is_expert_vm_enabled",
    "maybe_release_expert_vm_gpu_weights",
    "maybe_stage_expert_vm_gpu_weights",
    "parse_resident_layer_ids",
    "set_expert_vm_config_from_server_args",
]
