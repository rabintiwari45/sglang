from pathlib import Path
from typing import List

import importlib.util

import torch

_LOADED = False
_EMPTY_CPU_LONG: torch.Tensor | None = None


def _ensure_op_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    try:
        _ = torch.ops.sgl_kernel.expert_prefetch_copy
        _LOADED = True
        return
    except (AttributeError, RuntimeError):
        pass
    pkg_dir = Path(__file__).resolve().parent
    candidates = sorted(pkg_dir.glob("sgl_expert_prefetch_ops*.so"))
    if not candidates:
        raise ImportError(
            "expert_prefetch CUDA ops not built; run "
            "sgl-kernel/expert_prefetch_ext/build_and_install.py"
        )
    torch.ops.load_library(str(candidates[-1]))
    _ = torch.ops.sgl_kernel.expert_prefetch_copy
    _LOADED = True


def _empty_cpu_long() -> torch.Tensor:
    global _EMPTY_CPU_LONG
    if _EMPTY_CPU_LONG is None:
        _EMPTY_CPU_LONG = torch.empty(0, dtype=torch.long, pin_memory=True)
    return _EMPTY_CPU_LONG


def expert_prefetch_copy(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    expert_ids: torch.Tensor,
    copy_all: bool = False,
) -> None:
    """Batch H2D copy of MoE expert rows for router-predicted prefetch."""
    _ensure_op_loaded()
    torch.ops.sgl_kernel.expert_prefetch_copy.default(
        srcs, dsts, expert_ids, copy_all
    )


def expert_cache_copy(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    src_rows: torch.Tensor,
    dst_rows: torch.Tensor,
) -> None:
    """Row-remapped D2D/H2D batch copy on the current CUDA stream."""
    _ensure_op_loaded()
    torch.ops.sgl_kernel.expert_cache_copy.default(srcs, dsts, src_rows, dst_rows)


def expert_prefetch_launch(
    cpu_srcs: List[torch.Tensor],
    gpu_buf_dsts: List[torch.Tensor],
    cache_srcs: List[torch.Tensor],
    miss_ids: torch.Tensor,
    hit_expert_ids: torch.Tensor,
    hit_cache_slots: torch.Tensor,
    insert_expert_ids: torch.Tensor,
    insert_cache_slots: torch.Tensor,
) -> None:
    """One-shot cache hit restore + H2D misses + cache insert on current stream."""
    _ensure_op_loaded()
    torch.ops.sgl_kernel.expert_prefetch_launch.default(
        cpu_srcs,
        gpu_buf_dsts,
        cache_srcs,
        miss_ids,
        hit_expert_ids,
        hit_cache_slots,
        insert_expert_ids,
        insert_cache_slots,
    )


def expert_prefetch_unique_ids(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Deduplicate expert ids and drop out-of-range entries (CPU long tensor)."""
    _ensure_op_loaded()
    return torch.ops.sgl_kernel.expert_prefetch_unique_ids.default(ids, num_experts)


def expert_prefetch_unique_ids_list(ids: torch.Tensor, num_experts: int) -> List[int]:
    """Python list wrapper for small decode batches."""
    u = expert_prefetch_unique_ids(ids, num_experts)
    if u.numel() == 0:
        return []
    return [int(x) for x in u.tolist()]
