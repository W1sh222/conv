#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SIZE="${KERNEL_SIZE:-5}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-conv_kernel_5x5_ruler_mix_sparse_guarded_long_t07_96k128k_from_scratch_step4000}" \
  exec bash "${SCRIPT_DIR}/run_ruler_mix_sparse_guarded_128k_kernel_ablation.sh" "$@"
