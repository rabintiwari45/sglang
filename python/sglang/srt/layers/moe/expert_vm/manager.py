from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.moe.expert_vm.config import ExpertVMConfig, get_expert_vm_config
from sglang.srt.layers.moe.expert_vm.weights import get_expert_vm_param_names
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.utils import is_pin_memory_available

logger = logging.getLogger(__name__)


class ExpertVMManager:
    """Moves non-resident MoE expert weights to pinned CPU RAM after load."""

    def __init__(self, config: ExpertVMConfig):
        self.config = config
        self._offloaded_layers: List[int] = []

    def finalize_after_load(self, model: nn.Module) -> None:
        fused_moe_layers: List[Tuple[int, FusedMoE]] = []
        for module in model.modules():
            if isinstance(module, FusedMoE):
                fused_moe_layers.append((module.layer_id, module))

        fused_moe_layers.sort(key=lambda x: x[0])
        if not fused_moe_layers:
            logger.warning("[expert_vm] No FusedMoE layers found; nothing to offload.")
            return

        pin_memory = is_pin_memory_available()
        total_cpu_bytes = 0

        for layer_id, layer in fused_moe_layers:
            if self.config.is_resident_layer(layer_id):
                layer._expert_vm_offloaded = False  # type: ignore[attr-defined]
                # logger.info(
                #     "[expert_vm] Layer %d experts stay on GPU (resident).", layer_id
                # )
                continue

            self._offload_layer_experts(layer, layer_id, pin_memory)
            self._offloaded_layers.append(layer_id)
            for name in get_expert_vm_param_names(layer):
                cpu_buf = getattr(layer, f"expert_vm_{name}_cpu", None)
                if cpu_buf is not None:
                    total_cpu_bytes += cpu_buf.numel() * cpu_buf.element_size()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # logger.info(
        #     "[expert_vm] Offloaded %d MoE layer(s) to CPU RAM (~%.2f GiB pinned). "
        #     "Resident GPU expert layers: %s",
        #     len(self._offloaded_layers),
        #     total_cpu_bytes / (1024**3),
        #     sorted(self.config.resident_layer_ids),
        # )

    def _offload_layer_experts(
        self, layer: FusedMoE, layer_id: int, pin_memory: bool
    ) -> None:
        param_names = get_expert_vm_param_names(layer)
        if not param_names:
            logger.warning(
                "[expert_vm] Layer %d: no expert parameters found to offload.", layer_id
            )
            return

        anchor = getattr(layer, param_names[0])
        device = anchor.device
        if device.type == "cpu":
            self._finalize_cpu_resident_expert_params(
                layer, layer_id, pin_memory, param_names
            )
            return
        if device.type != "cuda":
            raise RuntimeError(
                f"[expert_vm] Layer {layer_id}: expected expert weights on CUDA or CPU "
                f"during finalize, got {device}."
            )

        for name in param_names:
            param = getattr(layer, name)
            cpu_tensor = _copy_param_to_pinned_cpu(param, pin_memory)
            layer.register_buffer(f"expert_vm_{name}_cpu", cpu_tensor, persistent=False)
            param.data = torch.empty(0, device=device, dtype=param.dtype)

        layer._expert_vm_offloaded = True  # type: ignore[attr-defined]
        # logger.debug(
        #     "[expert_vm] Layer %d expert weights moved to CPU (pinned=%s).",
        #     layer_id,
        #     pin_memory,
        # )

    def _finalize_cpu_resident_expert_params(
        self,
        layer: FusedMoE,
        layer_id: int,
        pin_memory: bool,
        param_names: List[str],
    ) -> None:
        """Expert weights were allocated on CPU during init; register buffers for staging."""
        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for name in param_names:
            param = getattr(layer, name)
            if pin_memory and not param.data.is_pinned():
                cpu_tensor = _copy_param_to_pinned_cpu(param, pin_memory)
            else:
                cpu_tensor = param.data.detach()
            layer.register_buffer(
                f"expert_vm_{name}_cpu", cpu_tensor, persistent=False
            )
            param.data = torch.empty(0, device=cuda_device, dtype=param.dtype)

        layer._expert_vm_offloaded = True  # type: ignore[attr-defined]
        # logger.debug(
        #     "[expert_vm] Layer %d expert weights finalized from CPU init (pinned=%s).",
        #     layer_id,
        #     pin_memory,
        # )


def _copy_param_to_pinned_cpu(
    param: torch.nn.Parameter, pin_memory: bool
) -> torch.Tensor:
    src = param.data.detach()
    cpu_data = torch.empty_strided(
        size=src.size(),
        stride=src.stride(),
        dtype=src.dtype,
        layout=src.layout,
        device="cpu",
        pin_memory=pin_memory,
    )
    cpu_data.copy_(src)
    return cpu_data


_manager: Optional[ExpertVMManager] = None


def get_expert_vm_manager() -> Optional[ExpertVMManager]:
    return _manager


def finalize_expert_vm_after_load(model: nn.Module) -> None:
    global _manager
    config = get_expert_vm_config()
    if config is None:
        _manager = None
        return
    _manager = ExpertVMManager(config)
    _manager.finalize_after_load(model)
    from sglang.srt.layers.moe.expert_vm.prefetch import register_expert_vm_sparse_blocks

    register_expert_vm_sparse_blocks(model)


def maybe_stage_expert_vm_gpu_weights(layer: FusedMoE) -> None:
    """Deprecated v1 full-layer staging; no-op when selective prefetch is used."""
    return


def maybe_release_expert_vm_gpu_weights(layer: FusedMoE) -> None:
    """Release selective GPU staging after MoE forward."""
    from sglang.srt.layers.moe.expert_vm.prefetch import expert_vm_release

    expert_vm_release(layer)
