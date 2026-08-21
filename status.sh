#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for instance in 8000 8001; do
  pid_file="$ROOT/run/sglang-$instance.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "instance=$instance status=running pid=$(cat "$pid_file")"
  else
    echo "instance=$instance status=stopped"
  fi
done

echo "--- GPU memory ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
