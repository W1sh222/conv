#!/usr/bin/env bash
set -euo pipefail

# Launch the eight Llama Conv RULER shards concurrently:
#   GPU 0: 32k-1, GPU 1: 32k-2,
#   GPU 2-4: 64k-1..3, GPU 5-7: 128k-1..3.
#
# Usage:
#   bash scripts/run_ruler_conv2_9_parallel.sh --weight PATH --topk 0.7
#   bash scripts/run_ruler_conv2_9_parallel.sh PATH 0.7

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/run_ruler_conv2_9_parallel.sh --weight PATH --topk RATIO [options]
   or: bash scripts/run_ruler_conv2_9_parallel.sh PATH RATIO [options]

Options:
  --weight PATH     Llama Conv .pt checkpoint (or initial_vertical_diag)
  --topk RATIO      block top-k ratio, for example 0.65 or 0.7
  --samples N       samples per task (default: 100)
  --stride N        block stride (default: 8)
  --log-dir PATH    per-GPU log directory
EOF
  exit 2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHT_PATH=""
TOPK=""
SAMPLES="${RULER_NUM_SAMPLES:-100}"
STRIDE="${STRIDE:-8}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/eval/RULER/scripts/parallel_logs/llama_conv2_9}"

if [[ $# -ge 2 && "$1" != -* ]]; then
  WEIGHT_PATH="$1"
  TOPK="$2"
  shift 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
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

[[ -n "${WEIGHT_PATH}" && -n "${TOPK}" ]] || usage
[[ "${TOPK}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || {
  echo "topk ratio must be in [0,1], got: ${TOPK}" >&2
  exit 2
}
if [[ "${WEIGHT_PATH}" != "initial_vertical_diag" && ! -f "${WEIGHT_PATH}" ]]; then
  echo "Llama Conv weight does not exist: ${WEIGHT_PATH}" >&2
  exit 1
fi

WEIGHT_TAG="$(basename -- "${WEIGHT_PATH}")"
WEIGHT_TAG="${WEIGHT_TAG%.pt}"
[[ -n "${WEIGHT_TAG}" ]] || WEIGHT_TAG="initial_vertical_diag"
mkdir -p "${LOG_DIR}"

RUNNERS=(
  run2.sh run3.sh
  run64_1.sh run64_2.sh run64_3.sh
  run128_1.sh run128_2.sh run128_3.sh
)
GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

echo "[parallel] model=llama3.1-8b-chat"
echo "[parallel] weight=${WEIGHT_PATH}"
echo "[parallel] topk=${TOPK} stride=${STRIDE} samples=${SAMPLES}"
echo "[parallel] output tag=${WEIGHT_TAG}"

for i in "${!RUNNERS[@]}"; do
  runner="${RUNNERS[$i]}"
  gpu="${GPUS[$i]}"
  log_file="${LOG_DIR}/gpu${gpu}_${runner%.sh}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export RULER_NUM_SAMPLES="${SAMPLES}"
    export RULER_RUN_TAG="${WEIGHT_TAG}"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    cd "${REPO_ROOT}/eval/RULER/scripts"
    exec bash "./${runner}" llama3.1-8b-chat synthetic \
      --stride "${STRIDE}" \
      --metric conv \
      --block_topk_ratio "${TOPK}" \
      --conv_weight_path "${WEIGHT_PATH}"
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

