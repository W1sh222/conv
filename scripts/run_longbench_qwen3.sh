#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${REPO_ROOT}/scripts/run_longbench_451.sh" qwen3 "${METHOD}" "$@"
