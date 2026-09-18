#!/usr/bin/env bash
# Run FWE block-swap Observation 1 and Observation 2 for a fixed q block
# across a pre-specified descending layer range. Every layer is retained.
set -u -o pipefail

usage() {
  echo "Usage: $0 MODEL OUTPUT_ROOT [START_LAYER] [END_LAYER] [HEAD] [SEQ_LENGTH] [SEED]" >&2
  echo "Example: $0 /models/Qwen3-8B output/ruler_observation/fwe_32k_seed42 16 0 8 32768 42" >&2
}

if [[ $# -lt 2 || $# -gt 7 ]]; then
  usage
  exit 2
fi

MODEL=$1
OUTPUT_ROOT=$2
START_LAYER=${3:-16}
END_LAYER=${4:-0}
HEAD=${5:-8}
SEQ_LENGTH=${6:-32768}
SEED=${7:-42}
QUERY_BLOCK=255
DATA_ROOT="${OUTPUT_ROOT}/_fwe_data"

if (( START_LAYER < END_LAYER )); then
  echo "START_LAYER must be >= END_LAYER for a descending sweep" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

# Generate and tokenize the FWE sample once. This directory is reused by all
# layers; only the layer-specific swap experiment reruns the model.
if [[ ! -f "${DATA_ROOT}/observation.jsonl" ]]; then
  python experiments/run_ruler_observation.py \
    --task fwe \
    --model "$MODEL" \
    --seq-length "$SEQ_LENGTH" \
    --num-samples 1 \
    --sample-index 0 \
    --seed "$SEED" \
    --layer "$START_LAYER" \
    --head "$HEAD" \
    --query-block "$QUERY_BLOCK" \
    --output "$DATA_ROOT" \
    --prepare-only
  if [[ $? -ne 0 ]]; then
    echo "FWE data preparation failed" >&2
    exit 1
  fi
fi

for layer in $(seq "$START_LAYER" -1 "$END_LAYER"); do
  LAYER_ROOT="${OUTPUT_ROOT}/layer_${layer}"
  SWAP_ROOT="${LAYER_ROOT}/swap"
  OBS_ROOT="${LAYER_ROOT}/observation2_line6"
  mkdir -p "$LAYER_ROOT"

  echo "===== FWE layer=${layer}, query_block=${QUERY_BLOCK} ====="

  if [[ ! -f "${SWAP_ROOT}/experiment.json" ]]; then
    python experiments/block_label_swap/run_experiment.py \
      --model "$MODEL" \
      --data "${DATA_ROOT}/observation.jsonl" \
      --sample-index 0 \
      --layer "$layer" \
      --head "$HEAD" \
      --query-block "$QUERY_BLOCK" \
      --ratio 0.65 \
      --stride 8 \
      --selector initial \
      --background sparse \
      --device-map auto \
      --dtype bfloat16 \
      --seed "$SEED" \
      --output "$SWAP_ROOT"
    if [[ $? -ne 0 ]]; then
      echo "Observation 1 failed at layer ${layer}; preserving later layers and continuing." >&2
      echo "observation1_failed" > "${LAYER_ROOT}/status.txt"
      continue
    fi
  fi

  if [[ ! -f "${OBS_ROOT}/summary.json" ]]; then
    python experiments2/run_observation2.py \
      --input "$SWAP_ROOT" \
      --output "$OBS_ROOT" \
      --line-radius 3
    if [[ $? -ne 0 ]]; then
      echo "Observation 2 failed at layer ${layer}; preserving the swap result and continuing." >&2
      echo "observation2_failed" > "${LAYER_ROOT}/status.txt"
      continue
    fi
  fi

  echo "Saved layer ${layer} to ${LAYER_ROOT}"
done

echo "Completed descending sweep. Results are under ${OUTPUT_ROOT}/layer_<N>/"
