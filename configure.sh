#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_PATH=${1:-${MODEL_PATH:-$ROOT/models/Qwen3.8-27B-FP8}}

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Model config not found: $MODEL_PATH/config.json" >&2
  echo "Run ./download-model.sh or pass the model directory to ./configure.sh." >&2
  exit 1
fi

MODEL_PATH=$(realpath "$MODEL_PATH")
API_KEY=${QWEN_API_KEY:-}

if [[ -z "$API_KEY" && -f "$ROOT/.env.8000" ]]; then
  API_KEY=$(sed -n 's/^QWEN_API_KEY=//p' "$ROOT/.env.8000" | head -n 1)
fi

if [[ -z "$API_KEY" ]]; then
  API_KEY=$(openssl rand -hex 32)
fi

write_env() {
  local file=$1

  cat >"$file" <<EOF
QWEN_API_KEY=$API_KEY
MODEL_PATH=$MODEL_PATH
MODEL_NAME=qwen3.8-27b
PORT=8001
TP_SIZE=8
EP_SIZE=2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENABLE_MTP=0
CONTEXT_LENGTH=131072
MAX_TOTAL_TOKENS=150000
CHUNKED_PREFILL_SIZE=8192
MEM_FRACTION_STATIC=0.90
MAX_RUNNING_REQUESTS=1
DISABLE_RADIX_CACHE=0
MAX_MAMBA_CACHE_SIZE=5
SPECULATIVE_ALGORITHM=DFLASH
DFLASH_DRAFT_MODEL_PATH=$ROOT/models/Qwen3.8-27B-DFlash2
DFLASH_BLOCK_SIZE=8
DFLASH_DRAFT_KV_CACHE_DTYPE=fp8_e4m3
DFLASH_DRAFT_WINDOW_SIZE=2048
MTP_DISABLE_DECODE_CUDA_GRAPH=0
REASONING_PARSER=qwen3
TOOL_CALL_PARSER=qwen3_coder
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
EOF
  chmod 600 "$file"
}

write_env "$ROOT/.env.8001"

echo "Created .env.8001 for the single 8-GPU TP8+EP2 deployment."
echo "The generated API key is stored only in the ignored .env files."
