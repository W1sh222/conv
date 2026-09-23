#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   bash scripts/run_longbench_conv.sh --conv_weight_path /path/to/weight.pt --block_topk_ratio 0.7
#   shorthand: bash scripts/run_longbench_conv.sh /path/to/weight.pt 0.7
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${REPO_ROOT}/scripts/run_longbench_451.sh" llama conv "$@"
