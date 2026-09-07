#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_efficiency.sh llama [attention_speedup.py args]
#   bash scripts/run_efficiency.sh qwen3 [attention_speedup.py args]
#   bash scripts/run_efficiency.sh both [attention_speedup.py args]
MODEL_KIND="${1:-llama}"
if [[ $# -gt 0 ]]; then shift; fi

case "${MODEL_KIND}" in
  llama|qwen3|both) ;;
  *) echo "Unsupported model: ${MODEL_KIND} (expected llama|qwen3|both)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${REPO_ROOT}"
mkdir -p output/efficiency

run_one() {
  python -u eval/efficiency/attention_speedup.py --model-kind "$1" "${@:2}"
}

if [[ "${MODEL_KIND}" == "both" ]]; then
  # Run sequentially so two 8B models never occupy the GPU simultaneously.
  run_one llama "$@"
  run_one qwen3 "$@"
else
  run_one "${MODEL_KIND}" "$@"
fi
