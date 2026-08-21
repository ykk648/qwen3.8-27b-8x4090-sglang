#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UV_BIN=${UV_BIN:-$(command -v uv || true)}

if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

if [[ -z "$UV_BIN" ]]; then
  echo "uv was not found; installing it with the Aliyun PyPI mirror."
  python3 -m pip install --user --index-url https://mirrors.aliyun.com/pypi/simple uv
  UV_BIN="$HOME/.local/bin/uv"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install a compatible NVIDIA driver first." >&2
  exit 1
fi

cd "$ROOT"
SGLANG_BUILD_RUST_EXTS=none \
  UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  "$UV_BIN" sync --python 3.12

echo "Environment ready: $ROOT/.venv"
