#!/usr/bin/env bash
set -euo pipefail

# Llama 128K counterpart of run64_3.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run128_common.sh" 3 "$@"
