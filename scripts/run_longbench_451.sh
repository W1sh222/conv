#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_longbench_451.sh llama conv
#   bash scripts/run_longbench_451.sh qwen3 xattn
MODEL_KIND="${1:-llama}"
if [[ $# -gt 0 ]]; then shift; fi
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

case "${MODEL_KIND}" in
  llama)
    MODEL_PATH="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct"
    ;;
  qwen3)
    MODEL_PATH="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B"
    ;;
  *)
    echo "unsupported model kind: ${MODEL_KIND}" >&2
    exit 2
    ;;
esac

case "${METHOD}" in
  xattn|conv|minference|flex|full) ;;
  *)
    echo "unsupported method: ${METHOD}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

DEFAULT_TASKS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum vcsum multi_news trec triviaqa samsum lsht lcc repobench-p"


TASKS="${LONGBENCH_TASKS:-${DEFAULT_TASKS}}"
STRIDE_VALUE="${STRIDE:-8}"
TOPK_VALUE="${BLOCK_TOPK_RATIO:-0.65}"
OUTPUT_TAG=""
if [[ "${MODEL_KIND}" == "llama" ]]; then
  OUTPUT_TAG="${LONGBENCH_RESULT_TAG:-conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_24k32k_bf16_step18000}"
fi
OUTPUT_TAG_ARGS=()
RESULTS_SUBDIR="${METHOD}"
if [[ -n "${OUTPUT_TAG}" ]]; then
  OUTPUT_TAG_ARGS+=(--result_tag "${OUTPUT_TAG}")
  RESULTS_SUBDIR="${OUTPUT_TAG}/${METHOD}"
fi

for TASK in ${TASKS}; do
  bash scripts/longbench.sh \
    "${MODEL_PATH}" "${TASK}" "${METHOD}" \
    --stride "${STRIDE_VALUE}" \
    --block_topk_ratio "${TOPK_VALUE}" \
    "${OUTPUT_TAG_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
done

MODEL_OUTPUT_NAME="$(basename "${MODEL_PATH}")"
python -u eval/LongBench/eval.py \
  --model "${MODEL_OUTPUT_NAME}" \
  --results_path "eval/LongBench/pred/${MODEL_OUTPUT_NAME}/${RESULTS_SUBDIR}/"
