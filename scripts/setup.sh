#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python"
export UV_CACHE_DIR="$PWD/.runtime/uv-cache"
mkdir -p .runtime
if [ ! -x .tools/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh -o .runtime/install-uv.sh
  UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh .runtime/install-uv.sh
fi
if [ ! -x .runtime/venv/bin/python ]; then
  .tools/uv venv --python 3.12 .runtime/venv
fi
.tools/uv pip install --python .runtime/venv/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.tools/uv pip install --python .runtime/venv/bin/python -r requirements.txt
.runtime/venv/bin/python scripts/fetch_reference.py
.runtime/venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
