#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_ruler_conv128_2.sh [xattn|conv|minference|flex|full] [extra RULER args]
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
case "${METHOD}" in
  xattn|conv|minference|flex|full) ;;
  *) echo "Unsupported method: ${METHOD} (expected xattn|conv|minference|flex|full)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"
DEFAULT_CONV_WEIGHT="${XATTN_ROOT}/conv_weights/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_24k32k_bf16_step18000.pt"
cd "${REPO_ROOT}/eval/RULER/scripts"

EXTRA_ARGS=("$@")
if [[ "${METHOD}" == "conv" ]]; then
  EXTRA_ARGS+=(--conv_weight_path "${CONV_WEIGHT_PATH:-${DEFAULT_CONV_WEIGHT}}")
fi

exec bash ./run128_2.sh llama3.1-8b-chat synthetic \
  --stride "${STRIDE:-8}" \
  --metric "${METHOD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO:-0.65}" \
  "${EXTRA_ARGS[@]}"
