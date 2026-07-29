from typing import List

import torch


def expert_prefetch_copy(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    expert_ids: torch.Tensor,
    copy_all: bool = False,
) -> None:
    """Batch H2D copy of MoE expert rows for router-predicted prefetch.

    ``srcs`` must be pinned CPU tensors; ``dsts`` are GPU buffers with matching
    shapes. When ``copy_all`` is True, copies each full tensor in one batch.
    Otherwise copies only the rows listed in ``expert_ids``.
    """
    torch.ops.sgl_kernel.expert_prefetch_copy.default(
        srcs, dsts, expert_ids, copy_all
    )
