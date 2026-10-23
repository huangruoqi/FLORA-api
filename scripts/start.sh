#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.runtime/reference:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/flora-api/huggingface}"
export FLORA_MODEL_CACHE="${FLORA_MODEL_CACHE:-$HOME/.cache/flora-api/models}"
exec .runtime/venv/bin/python -m uvicorn flora_api.app:app --host "${FLORA_HOST:-127.0.0.1}" --port "${FLORA_PORT:-8008}" --workers 1
