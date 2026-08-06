#!/usr/bin/env bash
# Install SGLang + sglang-kernel 0.4.3 + expert-prefetch JIT ops for the MoE
# CPU-offload / prefetch experiment (see python/sglang/srt/utils/expert_prefetch.py).
#
# Usage (from anywhere):
#   bash /workspace/sglang/install_experiment.sh
#
# Optional env:
#   SGLANG_ROOT=/path/to/sglang     repo root (default: directory containing this script)
#   VENV=/venv/main                 activate this venv if present
#   BUILD_SGL_KERNEL_FROM_SOURCE=1  build sgl-kernel from ./sgl-kernel instead of PyPI wheel
#   SKIP_APT=1                      skip apt-get (protobuf-compiler)
#   SKIP_RUST=1                     skip rustup (if cargo already on PATH)
#   SKIP_EXPERT_PREFETCH_EXT=1      skip JIT build of expert_cache_copy (slower D2D fallback only)

set -euo pipefail

SGLANG_ROOT="${SGLANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip}"

log() { echo "[install_experiment] $*"; }
die() { echo "[install_experiment] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Python / venv
# ---------------------------------------------------------------------------
if [[ -n "${VENV:-}" && -f "${VENV}/bin/activate" ]]; then
  log "Activating venv: ${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
elif [[ -f /venv/main/bin/activate ]]; then
  log "Activating default venv: /venv/main"
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

command -v "${PYTHON}" >/dev/null 2>&1 || die "python not found"
command -v "${PIP}" >/dev/null 2>&1 || die "pip not found"

[[ -d "${SGLANG_ROOT}/python" ]] || die "missing ${SGLANG_ROOT}/python (set SGLANG_ROOT?)"

# ---------------------------------------------------------------------------
# System packages (protobuf for gRPC / some deps; build tools for CUDA ext)
# ---------------------------------------------------------------------------
if [[ "${SKIP_APT:-0}" != "1" ]]; then
  log "Installing system packages (protobuf-compiler, build-essential, git)..."
  if command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
    sudo apt-get update
    sudo apt-get install -y protobuf-compiler build-essential git curl
  else
    apt-get update
    apt-get install -y protobuf-compiler build-essential git curl
  fi
else
  log "SKIP_APT=1 — not running apt-get"
fi

# ---------------------------------------------------------------------------
# Rust (sglang python package uses setuptools-rust for native helpers)
# ---------------------------------------------------------------------------
if [[ "${SKIP_RUST:-0}" != "1" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    log "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  else
    log "cargo already on PATH: $(command -v cargo)"
  fi
  if [[ -f "${HOME}/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.cargo/env"
  fi
  command -v cargo >/dev/null 2>&1 || die "cargo not available after rustup"
else
  log "SKIP_RUST=1 — assuming cargo is on PATH"
fi

# ---------------------------------------------------------------------------
# pip bootstrap
# ---------------------------------------------------------------------------
log "Upgrading pip..."
"${PIP}" install --upgrade pip

# PyTorch / CUDA stack: prefer the image venv versions; sglang pins torch==2.11.0
# in pyproject.toml and will install or align on editable install.
log "Ensuring build helpers for kernels..."
"${PIP}" install -U setuptools wheel setuptools-rust scikit-build-core ninja

# ---------------------------------------------------------------------------
# sglang-kernel 0.4.3
# ---------------------------------------------------------------------------
if [[ "${BUILD_SGL_KERNEL_FROM_SOURCE:-0}" == "1" ]]; then
  log "Building sglang-kernel 0.4.3 from source (requires CMake >= 3.31, CUDA)..."
  KERNEL_DIR="${SGLANG_ROOT}/sgl-kernel"
  [[ -d "${KERNEL_DIR}" ]] || die "missing ${KERNEL_DIR}"
  (
    cd "${KERNEL_DIR}"
    git submodule update --init --recursive 2>/dev/null || true
    make install-deps
    make build
  )
else
  log "Installing sglang-kernel==0.4.3 from PyPI..."
  "${PIP}" install "sglang-kernel==0.4.3"
fi

# ---------------------------------------------------------------------------
# SGLang (editable) — your experiment code lives here
# ---------------------------------------------------------------------------
log "Installing SGLang editable from ${SGLANG_ROOT}/python (this can take a while)..."
"${PIP}" install -v -e "${SGLANG_ROOT}/python"

# Re-pin kernel in case editable install pulled a different version
"${PIP}" install "sglang-kernel==0.4.3"

# ---------------------------------------------------------------------------
# Expert prefetch JIT extension (expert_cache_copy + updated wrapper)
# Required for GPU expert-cache D2D copies in expert_prefetch.py; H2D still
# works via the main wheel's expert_prefetch_copy if this step fails.
# ---------------------------------------------------------------------------
if [[ "${SKIP_EXPERT_PREFETCH_EXT:-0}" != "1" ]]; then
  EXT_DIR="${SGLANG_ROOT}/sgl-kernel/expert_prefetch_ext"
  if [[ -f "${EXT_DIR}/build_and_install.py" ]]; then
    log "Building expert_prefetch JIT extension (expert_cache_copy)..."
    if ! "${PYTHON}" "${EXT_DIR}/build_and_install.py"; then
      log "WARNING: expert_prefetch_ext build failed."
      log "         Prefetch will fall back to slow per-row copies for cache hits."
      log "         Check CUDA/nvcc and that torch sees GPU: ${PYTHON} -c 'import torch; print(torch.cuda.is_available())'"
    fi
  else
    log "WARNING: ${EXT_DIR}/build_and_install.py not found — skipping JIT extension"
  fi
else
  log "SKIP_EXPERT_PREFETCH_EXT=1 — skipping JIT extension"
fi

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
log "Verifying imports..."
"${PYTHON}" <<'PY'
import sys
import torch
import sglang
import sgl_kernel

print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("sglang", getattr(sglang, "__version__", "?"))
print("sgl_kernel", sgl_kernel.__file__)

from sgl_kernel.expert_prefetch import expert_prefetch_copy

print("expert_prefetch_copy OK")

try:
    from sgl_kernel.expert_prefetch import expert_cache_copy
    expert_cache_copy  # noqa: B018
    print("expert_cache_copy OK")
except Exception as e:
    print("expert_cache_copy MISSING (cache will use Python fallback):", e, file=sys.stderr)

try:
    from sglang.srt.utils.expert_prefetch import ExpertPrefetcher, get_expert_prefetcher
    print("ExpertPrefetcher OK")
except Exception as e:
    print("ExpertPrefetcher import failed:", e, file=sys.stderr)
    sys.exit(1)
PY

log "Done."
log "Run the experiment: cd ${SGLANG_ROOT} && ${PYTHON} main.py"
log "Or: ${PYTHON} ${SGLANG_ROOT}/bench_prefetch.py 32 10"
