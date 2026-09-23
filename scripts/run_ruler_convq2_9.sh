#!/usr/bin/env bash
set -euo pipefail

# Compatibility alias. The implementation lives in the explicit parallel
# launcher so both historical spellings use the same method/top-k interface.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_ruler_convq2_9_parallel.sh" "$@"
