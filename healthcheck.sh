#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTANCE=${1:-8000}
ENV_FILE="$ROOT/.env.$INSTANCE"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE." >&2
  exit 1
fi

source "$ENV_FILE"
curl --fail --silent --show-error \
  -H "Authorization: Bearer $QWEN_API_KEY" \
  "http://127.0.0.1:$PORT/v1/models"
echo
