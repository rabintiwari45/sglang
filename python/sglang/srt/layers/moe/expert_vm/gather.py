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
    topk_ids: torch.Tensor,
    active_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Map global expert ids to compact indices 0..K-1.

    `num_experts` is the static expert count for the layer; using it to size
    the lookup table avoids a `active_ids.max().item()` call, which would force
    a CPU<->GPU sync on the hot path every layer.
    """
    if active_ids.numel() == 0:
        return topk_ids.clone()
    lookup = torch.full(
        (num_experts,), -1, dtype=torch.int32, device=topk_ids.device
    )
    lookup[active_ids.to(torch.int64)] = torch.arange(
        active_ids.numel(), dtype=torch.int32, device=topk_ids.device
    )
    remapped = lookup[topk_ids.to(torch.int64)]
    return remapped


def expert_sets_match(
    active_a: torch.Tensor, active_b: torch.Tensor
) -> bool:
    """Compare two expert-id tensors without a GPU sync.

    Converts both tensors to sorted Python sets on CPU.  The tensors are
    tiny (≤8 int64s each) so the D2H transfer is negligible, and it avoids
    the torch.equal GPU sync that previously stalled the router step.
    """
    if active_a.numel() != active_b.numel():
        return False
    if active_a.numel() == 0:
        return True
    a = set(active_a.to(dtype=torch.int64, device="cpu").tolist())
    b = set(active_b.to(dtype=torch.int64, device="cpu").tolist())
    return a == b


def _expert_ids_as_sorted_list(ids: torch.Tensor) -> list[int]:
    if ids.numel() == 0:
        return []
    return sorted(int(x) for x in ids.to("cpu", torch.int64).tolist())


def expert_ids_as_sorted_list(ids: torch.Tensor) -> list[int]:
    """Public wrapper for logging and diagnostics."""
    return _expert_ids_as_sorted_list(ids)


def expert_set_lookahead_diff(
    predicted_ids: torch.Tensor, actual_ids: torch.Tensor
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Compare lookahead vs actual expert sets.

    Returns:
        predicted, actual, wrong_prefetch, missing, overlap (all sorted int lists).
        wrong_prefetch: predicted but not needed
        missing: needed but not predicted
    """
    predicted = _expert_ids_as_sorted_list(predicted_ids)
    actual = _expert_ids_as_sorted_list(actual_ids)
    pred_set = set(predicted)
    act_set = set(actual)
    wrong_prefetch = sorted(pred_set - act_set)
    missing = sorted(act_set - pred_set)
    overlap = sorted(pred_set & act_set)
    return predicted, actual, wrong_prefetch, missing, overlap


def gather_expert_rows_async(
    cpu_buffer: torch.Tensor,
    ids_list: list,
    dst: torch.Tensor,
    copy_stream: torch.cuda.Stream,
) -> None:
    """Copy selected expert rows from pinned CPU buffer to GPU tensor on copy_stream.

    Uses row-by-row views into the pinned cpu_buffer so each transfer is a
    true non-blocking DMA directly from pinned memory.  The previous approach
    called index_select() which produced a new **non-pinned** tensor; PyTorch
    then fell back to a synchronous staging copy (non-pinned → pinned →
    GPU), blocking the CPU thread for the entire ~8 ms copy duration and
    serialising the MoE compute that should have overlapped it.
    """
    with torch.cuda.stream(copy_stream):
        for i, eid in enumerate(ids_list):
            # cpu_buffer[eid] is a view into pinned memory → true async DMA
            dst[i].copy_(cpu_buffer[eid], non_blocking=True)


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
