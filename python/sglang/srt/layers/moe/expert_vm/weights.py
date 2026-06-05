from __future__ import annotations

from typing import List, Tuple

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

_UNQUANT_NAMES: Tuple[str, ...] = (
    "w13_weight",
    "w2_weight",
    "w13_weight_bias",
    "w2_weight_bias",
)

_GPTQ_MARLIN_NAMES: Tuple[str, ...] = (
    "w13_qweight",
    "w2_qweight",
    "w13_scales",
    "w2_scales",
    "w13_qzeros",
    "w2_qzeros",
    "w13_g_idx",
    "w2_g_idx",
    "w13_g_idx_sort_indices",
    "w2_g_idx_sort_indices",
)


def get_expert_vm_param_names(layer: FusedMoE) -> List[str]:
    """Return MoE expert parameter names for the layer's quantization method."""
    quant_method = layer.quant_method
    if quant_method is None:
        raise NotImplementedError(
            f"[expert_vm] Layer {layer.layer_id}: missing quant_method."
        )

    from sglang.srt.layers.quantization.gptq import GPTQMarlinMoEMethod
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    if isinstance(quant_method, UnquantizedFusedMoEMethod):
        candidate_names = _UNQUANT_NAMES
    elif isinstance(quant_method, GPTQMarlinMoEMethod):
        candidate_names = _GPTQ_MARLIN_NAMES
    else:
        raise NotImplementedError(
            f"[expert_vm] Layer {layer.layer_id}: unsupported MoE quant method "
            f"{type(quant_method).__name__}."
        )

    return [
        name
        for name in candidate_names
        if hasattr(layer, name) and getattr(layer, name) is not None
    ]
