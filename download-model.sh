#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.8-27B-FP8}
MODEL_DIR=${MODEL_DIR:-$ROOT/models/Qwen3.8-27B-FP8}
DRAFT_MODEL_ID=${DRAFT_MODEL_ID:-z-lab/Qwen3.8-27B-DFlash2}
DRAFT_MODEL_DIR=${DRAFT_MODEL_DIR:-$ROOT/models/Qwen3.8-27B-DFlash2}

if [[ ! -x "$ROOT/.venv/bin/modelscope" ]]; then
  echo "Run ./install.sh before downloading the model." >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"
"$ROOT/.venv/bin/modelscope" download \
  "$MODEL_ID" \
  --local-dir "$MODEL_DIR"

mkdir -p "$DRAFT_MODEL_DIR"
exec "$ROOT/.venv/bin/modelscope" download \
  "$DRAFT_MODEL_ID" \
  --local-dir "$DRAFT_MODEL_DIR"
