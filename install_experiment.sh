#!/usr/bin/env bash
# Minimal install for SGLang MoE expert-prefetch experiments on this Vast PyTorch image.
#
#   bash /workspace/sglang/install_experiment.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

echo "[install] upgrade pip"
pip install --upgrade pip

if ! command -v cargo >/dev/null 2>&1; then
  echo "[install] rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi

echo "[install] protobuf-compiler"
if command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
  sudo apt-get update
  sudo apt-get install -y protobuf-compiler
else
  apt-get update
  apt-get install -y protobuf-compiler
fi

echo "[install] sglang editable (${ROOT}/python)"
pip install -v -e "${ROOT}/python"

echo "[install] pin sglang-kernel 0.4.3 (matches pyproject; 0.4.5 breaks fp8 symbols)"
pip install "sglang-kernel==0.4.3"

echo "[install] pin kernels 0.14.1 (transformers 5.8.1 import compat)"
pip install "kernels==0.14.1"

if [[ -f "${ROOT}/sgl-kernel/expert_prefetch_ext/build_and_install.py" ]]; then
  echo "[install] expert_prefetch_copy JIT (batched H2D for prefetch)"
  python "${ROOT}/sgl-kernel/expert_prefetch_ext/build_and_install.py" || \
    echo "[install] WARNING: expert_prefetch_ext failed; prefetch still works, slower H2D"
fi

echo "[install] verify"
python -c "
import sglang
import sgl_kernel
from sgl_kernel.expert_prefetch import expert_prefetch_copy, expert_prefetch_launch
print('sglang OK', getattr(sglang, '__version__', '?'))
print('expert_prefetch_copy OK')
print('expert_prefetch_launch OK')
"

echo "[install] done — run: cd ${ROOT} && python main.py"
