#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_longbench_qwen3.sh conv --conv_weight_path /path/to/weight.pt --block_topk_ratio 0.7
#   shorthand: bash scripts/run_longbench_qwen3.sh conv /path/to/weight.pt 0.7
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${REPO_ROOT}/scripts/run_longbench_451.sh" qwen3 "${METHOD}" "$@"
