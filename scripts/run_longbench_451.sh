#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_longbench_451.sh llama conv
#   bash scripts/run_longbench_451.sh qwen3 conv --conv_weight_path /path/to/weight.pt --block_topk_ratio 0.7
#   shorthand for Conv: ... qwen3 conv /path/to/weight.pt 0.7
MODEL_KIND="${1:-llama}"
if [[ $# -gt 0 ]]; then shift; fi
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

# Conv convenience form: the first non-option argument is the .pt path and the
# second is the keep ratio. Flag form remains fully supported and is preferred
# when additional options are needed.
if [[ "${METHOD}" == "conv" && ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  if [[ "${EXTRA_ARGS[0]}" != -* ]]; then
    SHORT_WEIGHT="${EXTRA_ARGS[0]}"
    EXTRA_ARGS=(--conv_weight_path "${SHORT_WEIGHT}" "${EXTRA_ARGS[@]:1}")
    if [[ ${#EXTRA_ARGS[@]} -gt 2 && "${EXTRA_ARGS[2]}" != -* ]]; then
      SHORT_TOPK="${EXTRA_ARGS[2]}"
      EXTRA_ARGS=("${EXTRA_ARGS[@]:0:2}" --block_topk_ratio "${SHORT_TOPK}" "${EXTRA_ARGS[@]:3}")
    fi
  fi
fi

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

# Read an explicitly supplied Conv weight/result tag so each checkpoint gets a
# separate LongBench directory below the base model directory.
CONV_WEIGHT_ARG=""
EXPLICIT_RESULT_TAG=""
i=0
while (( i < ${#EXTRA_ARGS[@]} )); do
  ARG="${EXTRA_ARGS[$i]}"
  case "${ARG}" in
    --conv_weight_path|--conv-weight-path)
      i=$((i + 1))
      if (( i >= ${#EXTRA_ARGS[@]} )); then
        echo "${ARG} requires a path" >&2
        exit 2
      fi
      CONV_WEIGHT_ARG="${EXTRA_ARGS[$i]}"
      ;;
    --conv_weight_path=*|--conv-weight-path=*)
      CONV_WEIGHT_ARG="${ARG#*=}"
      ;;
    --block_topk_ratio)
      i=$((i + 1))
      if (( i >= ${#EXTRA_ARGS[@]} )); then
        echo "--block_topk_ratio requires a value" >&2
        exit 2
      fi
      TOPK_VALUE="${EXTRA_ARGS[$i]}"
      ;;
    --block_topk_ratio=*)
      TOPK_VALUE="${ARG#*=}"
      ;;
    --result_tag)
      i=$((i + 1))
      if (( i >= ${#EXTRA_ARGS[@]} )); then
        echo "--result_tag requires a value" >&2
        exit 2
      fi
      EXPLICIT_RESULT_TAG="${EXTRA_ARGS[$i]}"
      ;;
    --result_tag=*)
      EXPLICIT_RESULT_TAG="${ARG#*=}"
      ;;
  esac
  i=$((i + 1))
done

if [[ -n "${EXPLICIT_RESULT_TAG}" ]]; then
  OUTPUT_TAG="${EXPLICIT_RESULT_TAG}"
elif [[ -n "${LONGBENCH_RESULT_TAG:-}" ]]; then
  OUTPUT_TAG="${LONGBENCH_RESULT_TAG}"
elif [[ -n "${CONV_WEIGHT_ARG}" ]]; then
  OUTPUT_TAG="$(basename "${CONV_WEIGHT_ARG}")"
  OUTPUT_TAG="${OUTPUT_TAG%.pt}"
else
  OUTPUT_TAG="default_${METHOD}"
fi

OUTPUT_TAG_ARGS=()
RESULTS_SUBDIR="${METHOD}"
if [[ -n "${OUTPUT_TAG}" ]]; then
  OUTPUT_TAG_ARGS+=(--result_tag "${OUTPUT_TAG}")
  RESULTS_SUBDIR="${OUTPUT_TAG}/${METHOD}"
fi

echo "[LongBench] model_kind=${MODEL_KIND} model=${MODEL_PATH} method=${METHOD} stride=${STRIDE_VALUE} topk=${TOPK_VALUE}"
echo "[LongBench] conv_weight=${CONV_WEIGHT_ARG:-<model-default>}"
echo "[LongBench] output=eval/LongBench/pred/$(basename "${MODEL_PATH}")/${RESULTS_SUBDIR}"

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
