#!/usr/bin/env bash
set -euo pipefail

# 128K counterpart of runq64_1.sh (config_tasks3.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/runq128_common.sh" 1 "$@"
