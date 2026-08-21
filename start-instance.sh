#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTANCE=${1:-8000}
ENV_FILE=.env.$INSTANCE
PID_FILE="$ROOT/run/sglang-$INSTANCE.pid"
LOG_FILE="$ROOT/logs/sglang-$INSTANCE.log"

if [[ ! -f "$ROOT/$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Run ./configure.sh first." >&2
  exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/run"

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "Instance $INSTANCE is already running with PID $pid."
    exit 0
  fi
  rm -f "$PID_FILE"
fi

nohup env ENV_FILE="$ENV_FILE" "$ROOT/serve.sh" >"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$PID_FILE"
echo "Started instance $INSTANCE with PID $pid. Log: $LOG_FILE"
