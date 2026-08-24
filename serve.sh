#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${ENV_FILE:-.env.8000}

if [[ ! -f "$ROOT/$ENV_FILE" ]]; then
  echo "Configuration not found: $ROOT/$ENV_FILE" >&2
  echo "Run ./configure.sh first." >&2
  exit 1
fi

source "$ROOT/$ENV_FILE"

export PATH="$ROOT/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

speculative_args=()
capacity_args=()
cache_args=()
ssm_dtype=bfloat16

if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
  capacity_args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi
if [[ "${DISABLE_RADIX_CACHE:-1}" == "1" ]]; then
  cache_args+=(--disable-radix-cache)
fi

if [[ "${SPECULATIVE_ALGORITHM:-}" == "DFLASH" ]]; then
  speculative_args=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "${DFLASH_DRAFT_MODEL_PATH:?Set DFLASH_DRAFT_MODEL_PATH when SPECULATIVE_ALGORITHM=DFLASH}"
    --speculative-num-draft-tokens "${DFLASH_BLOCK_SIZE:-8}"
    --speculative-draft-kv-cache-dtype "${DFLASH_DRAFT_KV_CACHE_DTYPE:-fp8_e4m3}"
    --speculative-draft-window-size "${DFLASH_DRAFT_WINDOW_SIZE:-2048}"
  )
elif [[ "${ENABLE_MTP:-0}" == "1" ]]; then
  ssm_dtype=float32
  speculative_args=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --enable-linear-replayssm-spec
    --linear-replayssm-cache-len 32
  )
  if [[ "${MTP_DISABLE_DECODE_CUDA_GRAPH:-0}" == "1" ]]; then
    speculative_args+=(--disable-decode-cuda-graph)
  fi
fi

exec "$ROOT/.venv/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "${MODEL_NAME:-qwen3.8-27b}" \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --api-key "$QWEN_API_KEY" \
  --tp-size "${TP_SIZE:-4}" \
  --ep-size "${EP_SIZE:-1}" \
  --dcp-size "${DCP_SIZE:-1}" \
  --context-length "${CONTEXT_LENGTH:-262144}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-1}" \
  --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE:-1}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
  --disable-prefill-cuda-graph \
  --attention-backend triton \
  --quantization fp8 \
  --mamba-ssm-dtype "$ssm_dtype" \
  --triton-attention-reduce-in-fp32 \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
  "${capacity_args[@]}" \
  "${cache_args[@]}" \
  "${speculative_args[@]}"
