#!/usr/bin/env bash
set -euo pipefail

# Launch the eight Qwen3 RULER shards concurrently on GPUs 0-7.
# The shard layout matches the Llama parallel launcher.
#
# Usage:
#   bash scripts/run_ruler_convq2_9_parallel.sh --method conv --weight PATH --topk 0.7
#   bash scripts/run_ruler_convq2_9_parallel.sh --method flex --topk 0.7
#   bash scripts/run_ruler_convq2_9_parallel.sh --method minference
#   bash scripts/run_ruler_convq2_9_parallel.sh PATH 0.7  # shorthand for Conv

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/run_ruler_convq2_9_parallel.sh --method METHOD [options]
   or: bash scripts/run_ruler_convq2_9_parallel.sh PATH RATIO [options]

Options:
  --method METHOD   conv, xattn, flex, minference, or full (default: conv)
  --weight PATH     Qwen3 Conv .pt checkpoint (required for conv)
  --topk RATIO      block top-k ratio for conv/xattn/flex
  --samples N       samples per task (default: 100)
  --stride N        block stride (default: 8)
  --log-dir PATH    per-GPU log directory
EOF
  exit 2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHT_PATH=""
TOPK=""
METHOD="conv"
SAMPLES="${RULER_NUM_SAMPLES:-100}"
STRIDE="${STRIDE:-8}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/eval/RULER/scripts/parallel_logs/qwen_conv2_9}"

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
if [[ "${METHOD}" == "conv" || "${METHOD}" == "xattn" || "${METHOD}" == "flex" ]]; then
  TOPK="${TOPK:-0.65}"
  [[ "${TOPK}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0*)?)$ ]] || {
    echo "topk ratio must be in [0,1], got: ${TOPK}" >&2
    exit 2
  }
fi
if [[ "${METHOD}" == "conv" && ! -f "${WEIGHT_PATH}" ]]; then
  echo "Qwen3 Conv weight does not exist: ${WEIGHT_PATH}" >&2
  exit 1
fi

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
  runq2.sh runq3.sh
  runq64_1.sh runq64_2.sh runq64_3.sh
  runq128_1.sh runq128_2.sh runq128_3.sh
)
GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

echo "[parallel] model=qwen3-8b"
echo "[parallel] method=${METHOD}"
echo "[parallel] weight=${WEIGHT_PATH:-<not-used>}"
echo "[parallel] topk=${TOPK:-<not-used>} stride=${STRIDE} samples=${SAMPLES}"
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
    RUN_ARGS=(
      --stride "${STRIDE}"
      --metric "${METHOD}"
    )
    if [[ "${METHOD}" != "minference" && "${METHOD}" != "full" ]]; then
      RUN_ARGS+=(--block_topk_ratio "${TOPK}")
    fi
    if [[ "${METHOD}" == "conv" ]]; then
      RUN_ARGS+=(--conv_weight_path "${WEIGHT_PATH}")
    fi
    exec bash "./${runner}" qwen3-8b synthetic \
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
