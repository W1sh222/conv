#!/usr/bin/env bash
# Aggregate already-complete VT layer swap sweeps into Observation 3.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 VT_LAYER_ROOT [OUTPUT]" >&2
  echo "Example: $0 output/ruler_observation/vt_32k_seed42_layers" >&2
  exit 2
fi

ROOT=$1
OUTPUT=${2:-"${ROOT}/observation3_signed7x7"}

python experiments3/run_observation3.py \
  --input "${ROOT}"/layer_*/swap \
  --output "$OUTPUT" \
  --kernel-size 7 \
  --folds 5 \
  --purge-radius 6 \
  --ridge-alpha 0.1 \
  --permutations 2000 \
  --bootstrap-samples 10000 \
  --seed 42

