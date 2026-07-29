#!/usr/bin/env python3
"""JIT-build expert_prefetch_copy and install it into the active sgl_kernel package."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import sgl_kernel
import torch
from torch.utils.cpp_extension import load


def main() -> int:
    here = Path(__file__).resolve().parent
    src = here / "expert_prefetch_copy.cu"
    build_dir = here / "build"
    build_dir.mkdir(exist_ok=True)

    print(f"Building expert_prefetch_copy from {src}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")

    # TORCH_LIBRARY_* registration is not a Python module — load as a
    # plain shared library so ops register into torch.ops.sgl_kernel.
    load(
        name="sgl_expert_prefetch_ops",
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=True,
    )
    print("Built shared library")

    # Sanity: op registered under torch.ops.sgl_kernel
    op = torch.ops.sgl_kernel.expert_prefetch_copy
    print("Registered op:", op)

    pkg_dir = Path(sgl_kernel.__file__).resolve().parent
    wrapper_src = (
        Path(__file__).resolve().parents[1] / "python" / "sgl_kernel" / "expert_prefetch.py"
    )
    if not wrapper_src.exists():
        # Fallback: write a minimal wrapper next to this script.
        wrapper_src = here / "expert_prefetch.py"
        wrapper_src.write_text(
            '''from typing import List

import torch


def expert_prefetch_copy(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    expert_ids: torch.Tensor,
    copy_all: bool = False,
) -> None:
    """Batch H2D copy of MoE expert rows for router-predicted prefetch."""
    torch.ops.sgl_kernel.expert_prefetch_copy.default(
        srcs, dsts, expert_ids, copy_all
    )
'''
        )

    dst_wrapper = pkg_dir / "expert_prefetch.py"
    shutil.copy2(wrapper_src, dst_wrapper)
    print(f"Installed wrapper -> {dst_wrapper}")

    # Ensure the compiled .so stays loadable.  Copy into package and add a
    # tiny import hook in expert_prefetch.py that loads it if needed.
    so_candidates = list(build_dir.glob("sgl_expert_prefetch_ops*.so"))
    if not so_candidates:
        print("WARNING: could not find built .so in", build_dir, file=sys.stderr)
        return 1
    so_src = so_candidates[0]
    so_dst = pkg_dir / so_src.name
    shutil.copy2(so_src, so_dst)
    print(f"Installed extension -> {so_dst}")

    # Rewrite wrapper to load the .so before calling the op.
    dst_wrapper.write_text(
        f'''from typing import List

import torch


def _ensure_op_loaded() -> None:
    if hasattr(torch.ops.sgl_kernel, "expert_prefetch_copy"):
        return
    from pathlib import Path
    import importlib.util

    so = Path(__file__).resolve().parent / "{so_src.name}"
    spec = importlib.util.spec_from_file_location("sgl_expert_prefetch_ops", so)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load expert_prefetch extension from {{so}}")
    importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))  # type: ignore[arg-type]


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
'''
    )
    # Fix the loader: exec_module needs the module object.
    dst_wrapper.write_text(
        f'''from typing import List
from pathlib import Path
import importlib.util

import torch

_LOADED = False


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
    so = Path(__file__).resolve().parent / "{so_src.name}"
    torch.ops.load_library(str(so))
    _ = torch.ops.sgl_kernel.expert_prefetch_copy
    _LOADED = True


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
'''
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
