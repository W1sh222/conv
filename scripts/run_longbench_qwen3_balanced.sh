#!/usr/bin/env bash
set -euo pipefail

# LongBench companion for the balanced Qwen3 checkpoint. Native RoPE is the
# default because most LongBench prompts are short enough that static YaRN can
# reduce base-model quality. Set QWEN_LONG_ROPE_MODE=yarn for >40K-only runs.
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
case "${METHOD}" in
  xattn|conv|minference|flex|full) ;;
  *) echo "unsupported method: ${METHOD}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"
DEFAULT_NATIVE_WEIGHT="${XATTN_ROOT}/qwen_weights/conv_qwen3_t065_64k128k_balanced_v3/conv_kernel_7x7_qwen3_t065_longbench_native_8k64k_s8_bf16_ema.pt"
DEFAULT_YARN_WEIGHT="${XATTN_ROOT}/qwen_weights/conv_qwen3_t065_64k128k_balanced_v3/conv_kernel_7x7_qwen3_t065_balanced_32k128k_s8_yarn4_bf16_ema.pt"
ROPE_MODE="${QWEN_LONG_ROPE_MODE:-native}"

EXTRA_ARGS=("$@")
case "${ROPE_MODE}" in
  native)
    DEFAULT_WEIGHT="${DEFAULT_NATIVE_WEIGHT}"
    EXTRA_ARGS+=(--rope_scaling_type none)
    ;;
  yarn)
    DEFAULT_WEIGHT="${DEFAULT_YARN_WEIGHT}"
    EXTRA_ARGS+=(
      --rope_scaling_type yarn
      --rope_factor 4.0
      --rope_original_max_position_embeddings 32768
      --max_position_embeddings_override 131072
    )
    ;;
  *)
    echo "QWEN_LONG_ROPE_MODE must be native or yarn, got: ${ROPE_MODE}" >&2
    exit 2
    ;;
esac

if [[ "${METHOD}" == "conv" ]]; then
  EXTRA_ARGS+=(--conv_weight_path "${CONV_WEIGHT_PATH:-${DEFAULT_WEIGHT}}")
fi

export BLOCK_TOPK_RATIO="${BLOCK_TOPK_RATIO:-0.65}"
export STRIDE="${STRIDE:-8}"
exec bash "${REPO_ROOT}/scripts/run_longbench_qwen3.sh" "${METHOD}" "${EXTRA_ARGS[@]}"
