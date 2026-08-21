#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTANCE=${1:-8000}
PID_FILE="$ROOT/run/sglang-$INSTANCE.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Instance $INSTANCE has no PID file."
  exit 0
fi

pid=$(cat "$PID_FILE")
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in $(seq 1 60); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "Instance $INSTANCE did not stop within 60 seconds; PID file was kept." >&2
    exit 1
  fi
fi

rm -f "$PID_FILE"
echo "Stopped instance $INSTANCE."
