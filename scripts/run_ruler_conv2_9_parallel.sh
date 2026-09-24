#!/usr/bin/env bash
set -euo pipefail

# Launch the eight Llama RULER shards concurrently:
#   GPU 0: 32k-1, GPU 1: 32k-2,
#   GPU 2-4: 64k-1..3, GPU 5-7: 128k-1..3.
#
# Usage:
#   bash scripts/run_ruler_conv2_9_parallel.sh --method conv --weight PATH --topk 0.7
#   bash scripts/run_ruler_conv2_9_parallel.sh --method flex --flex-gamma 0.9 --flex-tau 0.1
#   bash scripts/run_ruler_conv2_9_parallel.sh --method minference
#   bash scripts/run_ruler_conv2_9_parallel.sh PATH 0.7  # shorthand for Conv

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/run_ruler_conv2_9_parallel.sh --method METHOD [options]
   or: bash scripts/run_ruler_conv2_9_parallel.sh PATH RATIO [options]

Options:
  --method METHOD   conv, xattn, flex, minference, or full (default: conv)
  --weight PATH     Llama Conv .pt checkpoint (required for conv)
  --topk RATIO      block top-k ratio for conv/xattn (ignored by flex)
  --flex-gamma X    original Flex attention-mass coverage (default: 0.9)
  --flex-tau X      original Flex JS-divergence threshold (default: 0.1)
  --minference-vertical N  MInference vertical budget (default: 1000)
  --minference-slash N     MInference slash budget (default: 6096)
  --samples N       samples per task (default: 100)
  --stride N        block stride (default: 8)
  --log-dir PATH    per-GPU log directory
EOF
  exit 2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHT_PATH=""
TOPK=""
FLEX_GAMMA="${FLEX_GAMMA:-0.9}"
FLEX_TAU="${FLEX_TAU:-0.1}"
METHOD="conv"
MINFERENCE_VERTICAL_SIZE="${MINFERENCE_VERTICAL_SIZE:-1000}"
MINFERENCE_SLASH_SIZE="${MINFERENCE_SLASH_SIZE:-6096}"
SAMPLES="${RULER_NUM_SAMPLES:-100}"
STRIDE="${STRIDE:-8}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/eval/RULER/scripts/parallel_logs/llama_conv2_9}"

if [[ $# -gt 0 && "$1" != -* ]]; then
  case "$1" in
    conv|xattn|flex|minference|full)
      METHOD="$1"
      shift
      if [[ "$METHOD" == "conv" && $# -ge 2 && "$1" != -* ]]; then
        WEIGHT_PATH="$1"
        TOPK="$2"
        shift 2
      elif [[ "$METHOD" != "minference" && "$METHOD" != "full" && $# -ge 1 && "$1" != -* ]]; then
        TOPK="$1"
        shift
      fi
      ;;
    *)
      if [[ $# -ge 2 ]]; then
        WEIGHT_PATH="$1"
        TOPK="$2"
        shift 2
      else
        usage
      fi
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      [[ $# -ge 2 ]] || usage
      METHOD="$2"
      shift 2
      ;;
    --weight|--weight_path|--conv_weight_path)
      [[ $# -ge 2 ]] || usage
      WEIGHT_PATH="$2"
      shift 2
      ;;
    --topk|--block_topk_ratio)
      [[ $# -ge 2 ]] || usage
      TOPK="$2"
      shift 2
      ;;
    --flex-gamma|--flex_gamma)
      [[ $# -ge 2 ]] || usage
      FLEX_GAMMA="$2"
      shift 2
      ;;
    --flex-tau|--flex_tau)
      [[ $# -ge 2 ]] || usage
      FLEX_TAU="$2"
      shift 2
      ;;
    --minference-vertical|--minference_vertical_size)
      [[ $# -ge 2 ]] || usage
      MINFERENCE_VERTICAL_SIZE="$2"
      shift 2
      ;;
    --minference-slash|--minference_slash_size)
      [[ $# -ge 2 ]] || usage
      MINFERENCE_SLASH_SIZE="$2"
      shift 2
      ;;
    --samples)
      [[ $# -ge 2 ]] || usage
      SAMPLES="$2"
      shift 2
      ;;
    --stride)
      [[ $# -ge 2 ]] || usage
      STRIDE="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || usage
      LOG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      ;;
  esac
done

case "${METHOD}" in
  conv|xattn|flex|minference|full) ;;
  *) echo "unsupported method: ${METHOD}" >&2; usage ;;
esac

if [[ "${METHOD}" == "conv" && -z "${WEIGHT_PATH}" ]]; then
  echo "--weight is required for method=conv" >&2
  usage
fi
if [[ "${METHOD}" == "conv" || "${METHOD}" == "xattn" ]]; then
  TOPK="${TOPK:-0.65}"
  [[ "${TOPK}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0*)?)$ ]] || {
    echo "topk ratio must be in [0,1], got: ${TOPK}" >&2
    exit 2
  }
fi
if [[ "${METHOD}" == "conv" && "${WEIGHT_PATH}" != "initial_vertical_diag" && ! -f "${WEIGHT_PATH}" ]]; then
  echo "Llama Conv weight does not exist: ${WEIGHT_PATH}" >&2
  exit 1
fi
[[ "${MINFERENCE_VERTICAL_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MInference vertical budget must be a positive integer" >&2
  exit 2
}
[[ "${MINFERENCE_SLASH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MInference slash budget must be a positive integer" >&2
  exit 2
}
awk "BEGIN { if (!(${FLEX_GAMMA} > 0 && ${FLEX_GAMMA} <= 1)) exit 1 }" || {
  echo "Flex gamma must be in (0,1], got: ${FLEX_GAMMA}" >&2
  exit 2
}
awk "BEGIN { if (!(${FLEX_TAU} >= 0)) exit 1 }" || {
  echo "Flex tau must be non-negative, got: ${FLEX_TAU}" >&2
  exit 2
}

if [[ -n "${RULER_RUN_TAG:-}" ]]; then
  WEIGHT_TAG="${RULER_RUN_TAG}"
elif [[ "${METHOD}" == "conv" ]]; then
  WEIGHT_TAG="$(basename -- "${WEIGHT_PATH}")"
  WEIGHT_TAG="${WEIGHT_TAG%.pt}"
else
  WEIGHT_TAG="${METHOD}"
fi
mkdir -p "${LOG_DIR}"

RUNNERS=(
  run2.sh run3.sh
  run64_1.sh run64_2.sh run64_3.sh
  run128_1.sh run128_2.sh run128_3.sh
)
GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

echo "[parallel] model=llama3.1-8b-chat"
echo "[parallel] method=${METHOD}"
echo "[parallel] weight=${WEIGHT_PATH:-<not-used>}"
if [[ "${METHOD}" == "flex" ]]; then
  echo "[parallel] flex_gamma=${FLEX_GAMMA} flex_tau=${FLEX_TAU} stride=<not-used> samples=${SAMPLES}"
else
  echo "[parallel] topk=${TOPK:-<not-used>} stride=${STRIDE} samples=${SAMPLES}"
fi
echo "[parallel] minference_vertical=${MINFERENCE_VERTICAL_SIZE} minference_slash=${MINFERENCE_SLASH_SIZE}"
echo "[parallel] output tag=${WEIGHT_TAG}"

for i in "${!RUNNERS[@]}"; do
  runner="${RUNNERS[$i]}"
  gpu="${GPUS[$i]}"
  log_file="${LOG_DIR}/gpu${gpu}_${runner%.sh}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export RULER_NUM_SAMPLES="${SAMPLES}"
    export RULER_RUN_TAG="${WEIGHT_TAG}"
    export MINFERENCE_VERTICAL_SIZE="${MINFERENCE_VERTICAL_SIZE}"
    export MINFERENCE_SLASH_SIZE="${MINFERENCE_SLASH_SIZE}"
    export FLEX_GAMMA="${FLEX_GAMMA}"
    export FLEX_TAU="${FLEX_TAU}"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    cd "${REPO_ROOT}/eval/RULER/scripts"
    RUN_ARGS=(
      --stride "${STRIDE}"
      --metric "${METHOD}"
    )
    if [[ "${METHOD}" == "conv" || "${METHOD}" == "xattn" ]]; then
      RUN_ARGS+=(--block_topk_ratio "${TOPK}")
    fi
    if [[ "${METHOD}" == "conv" ]]; then
      RUN_ARGS+=(--conv_weight_path "${WEIGHT_PATH}")
    fi
    exec bash "./${runner}" llama3.1-8b-chat synthetic \
      "${RUN_ARGS[@]}"
  ) >"${log_file}" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  echo "[launch] gpu=${gpu} runner=${runner} pid=${pid} log=${log_file}"
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[done] gpu=${GPUS[$i]} runner=${RUNNERS[$i]}"
  else
    echo "[failed] gpu=${GPUS[$i]} runner=${RUNNERS[$i]} (see ${LOG_DIR}/gpu${GPUS[$i]}_${RUNNERS[$i]%.sh}.log)" >&2
    FAILED=1
  fi
done

exit "${FAILED}"
