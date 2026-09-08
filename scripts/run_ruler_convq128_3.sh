#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_ruler_convq128_3.sh [xattn|conv|minference|flex|full] [extra RULER args]
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
case "${METHOD}" in
  xattn|conv|minference|flex|full) ;;
  *) echo "Unsupported method: ${METHOD} (expected xattn|conv|minference|flex|full)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"
DEFAULT_CONV_WEIGHT="${XATTN_ROOT}/qwen_weights/conv_qwen3_t065_64k128k_balanced_v3/conv_kernel_7x7_qwen3_t065_balanced_32k128k_s8_yarn4_bf16_ema.pt"
cd "${REPO_ROOT}/eval/RULER/scripts"

EXTRA_ARGS=("$@")
if [[ "${METHOD}" == "conv" ]]; then
  EXTRA_ARGS+=(--conv_weight_path "${CONV_WEIGHT_PATH:-${DEFAULT_CONV_WEIGHT}}")
fi

exec bash ./runq128_3.sh qwen3-8b synthetic \
  --stride "${STRIDE:-8}" \
  --metric "${METHOD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO:-0.65}" \
  "${EXTRA_ARGS[@]}"
