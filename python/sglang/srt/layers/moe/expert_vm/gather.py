from __future__ import annotations

import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


def get_active_expert_ids(topk_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return sorted unique expert ids referenced by topk_ids (GPU tensor)."""
    flat = topk_ids.reshape(-1)
    valid = flat[flat >= 0]
    if valid.numel() == 0:
        active = torch.zeros(0, dtype=torch.int64, device=topk_ids.device)
    else:
        active = torch.unique(valid)
    return active, int(active.numel())


def remap_topk_ids(
    topk_ids: torch.Tensor, active_ids: torch.Tensor
) -> torch.Tensor:
    """Map global expert ids to compact indices 0..K-1."""
    if active_ids.numel() == 0:
        return topk_ids.clone()
    # active_ids sorted from unique
    max_id = int(active_ids.max().item()) if active_ids.numel() else 0
    lookup = torch.full(
        (max_id + 1,), -1, dtype=torch.int32, device=topk_ids.device
    )
    lookup[active_ids.to(torch.int64)] = torch.arange(
        active_ids.numel(), dtype=torch.int32, device=topk_ids.device
    )
    remapped = lookup[topk_ids.to(torch.int64)]
    return remapped


def expert_sets_match(
    active_a: torch.Tensor, active_b: torch.Tensor
) -> bool:
    if active_a.numel() != active_b.numel():
        return False
    if active_a.numel() == 0:
        return True
    return torch.equal(
        torch.sort(active_a)[0].to(torch.int64),
        torch.sort(active_b)[0].to(torch.int64),
    )


def gather_expert_rows_async(
    cpu_buffer: torch.Tensor,
    active_ids: torch.Tensor,
    dst: torch.Tensor,
    copy_stream: torch.cuda.Stream,
) -> None:
    """Copy selected expert rows from pinned CPU buffer to GPU tensor on copy_stream."""
    with torch.cuda.stream(copy_stream):
        if active_ids.device.type != "cpu":
            ids_cpu = active_ids.to(dtype=torch.int64, device="cpu")
        else:
            ids_cpu = active_ids.to(dtype=torch.int64)
        dst.copy_(cpu_buffer.index_select(0, ids_cpu), non_blocking=True)


def allocate_compact_gpu_tensors(
    layer,
    param_names: List[str],
    active_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate empty GPU tensors with K expert rows for each weight name."""
    k = int(active_ids.numel())
    quant_method = getattr(layer, "quant_method", None)
    act_dtype = getattr(quant_method, "_params_dtype", None)
    scale_names = frozenset({"w13_scales", "w2_scales"})
    out: dict[str, torch.Tensor] = {}
    for name in param_names:
        cpu_buf = getattr(layer, f"expert_vm_{name}_cpu")
        if cpu_buf is None:
            continue
        rest = cpu_buf.shape[1:]
        dtype = act_dtype if act_dtype is not None and name in scale_names else cpu_buf.dtype
        out[name] = torch.empty(
            (k, *rest), dtype=dtype, device=device
        )
    return out
