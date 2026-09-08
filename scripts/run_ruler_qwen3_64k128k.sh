#!/usr/bin/env bash
set -euo pipefail

# Evaluate Qwen3 with the same static YaRN coordinates used by the 64K-128K
# trainer. Usage: bash scripts/run_ruler_qwen3_64k128k.sh [conv|xattn|full]
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
case "${METHOD}" in
  xattn|conv|minference|flex|full) ;;
  *) echo "unsupported method: ${METHOD}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"
DEFAULT_WEIGHT="${XATTN_ROOT}/qwen_weights/conv_qwen3_t065_64k128k_balanced_v3/conv_kernel_7x7_qwen3_t065_balanced_32k128k_s8_yarn4_bf16_ema.pt"

export RULER_SEQ_LENGTHS="${RULER_SEQ_LENGTHS:-65536 98304 131072}"
export RULER_TASKS="${RULER_TASKS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2}"
export RULER_RUN_TAG="${RULER_RUN_TAG:-qwen3_t065_yarn4_64k128k_}"
export ROPE_SCALING_TYPE=yarn
export ROPE_FACTOR=4.0
export ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=32768
export MAX_POSITION_EMBEDDINGS_OVERRIDE=131072
export BLOCK_TOPK_RATIO="${BLOCK_TOPK_RATIO:-0.65}"
export STRIDE="${STRIDE:-8}"

EXTRA_ARGS=("$@")
if [[ "${METHOD}" == "conv" ]]; then
  EXTRA_ARGS+=(--conv_weight_path "${CONV_WEIGHT_PATH:-${DEFAULT_WEIGHT}}")
fi

exec bash "${REPO_ROOT}/scripts/run_ruler_convq7.sh" "${METHOD}" "${EXTRA_ARGS[@]}"
