#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install a current NVIDIA driver first." >&2
  exit 1
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "ERROR: expected an existing virtual environment at $ROOT/.venv" >&2
  echo "Create it with: python3 -m venv .venv" >&2
  exit 1
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Current official PyTorch CUDA 13.0 wheels. The RTX 3060 Ti is Ampere (SM 8.6)
# and the installed NVIDIA driver supports CUDA 13.0. These wheels bring their
# own CUDA user-space libraries, so the host CUDA toolkit version is irrelevant.
python -m pip install --upgrade torch==2.14.0 torchvision==0.29.0 \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple
if python -m pip show cupy-cuda12x >/dev/null 2>&1; then
  python -m pip uninstall -y cupy-cuda12x
fi
python -m pip install --upgrade -e ".[cuda,dev]"

python -m sts2.check_install
pytest -q

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Smoke benchmark: ./scripts/run_a0.sh"
