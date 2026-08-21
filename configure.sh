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
  local port=$2
  local devices=$3

  cat >"$file" <<EOF
QWEN_API_KEY=$API_KEY
MODEL_PATH=$MODEL_PATH
MODEL_NAME=qwen3.8-27b
PORT=$port
TP_SIZE=4
EP_SIZE=1
CUDA_VISIBLE_DEVICES=$devices
ENABLE_MTP=1
CONTEXT_LENGTH=262144
MAX_TOTAL_TOKENS=300000
CHUNKED_PREFILL_SIZE=8192
MEM_FRACTION_STATIC=0.85
MAX_RUNNING_REQUESTS=1
MTP_DISABLE_DECODE_CUDA_GRAPH=0
REASONING_PARSER=qwen3
TOOL_CALL_PARSER=qwen3_coder
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
EOF
  chmod 600 "$file"
}

write_env "$ROOT/.env.8000" 8000 0,1,2,3
write_env "$ROOT/.env.8001" 8001 4,5,6,7

echo "Created .env.8000 for GPUs 0-3 and .env.8001 for GPUs 4-7."
echo "The generated API key is stored only in the ignored .env files."
