#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$ROOT/stop-instance.sh" 8000
"$ROOT/stop-instance.sh" 8001
