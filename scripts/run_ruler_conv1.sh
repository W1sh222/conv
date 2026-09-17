#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_ruler_conv1.sh [xattn|conv|minference|flex|full] [extra RULER args]
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
case "${METHOD}" in
  xattn|conv|minference|flex|full) ;;
  *) echo "Unsupported method: ${METHOD} (expected xattn|conv|minference|flex|full)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MODEL_ATTENTION_IMPLEMENTATION="${MODEL_ATTENTION_IMPLEMENTATION:-sdpa}"
export RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-100}"
cd "${REPO_ROOT}/eval/RULER/scripts"

EXTRA_ARGS=("$@")
if [[ "${METHOD}" == "conv" ]]; then
  # Default is the deterministic initial kernel; a final *.pt argument or
  # --conv_weight_path overrides it and names the result directory.
  source "${REPO_ROOT}/scripts/resolve_conv_eval_args.sh"
  resolve_conv_eval_args "initial_vertical_diag"
else
  export RULER_RUN_TAG="other_method"
fi

exec bash ./run1.sh llama3.1-8b-chat synthetic \
  --stride "${STRIDE:-8}" \
  --metric "${METHOD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO:-0.65}" \
  "${EXTRA_ARGS[@]}"
