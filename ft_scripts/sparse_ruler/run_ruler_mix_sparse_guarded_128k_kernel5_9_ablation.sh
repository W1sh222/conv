#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[ablation] starting kernel=5"
bash "${SCRIPT_DIR}/run_ruler_mix_sparse_guarded_128k_kernel5_ablation.sh" "$@"

echo "[ablation] starting kernel=9"
bash "${SCRIPT_DIR}/run_ruler_mix_sparse_guarded_128k_kernel9_ablation.sh" "$@"

echo "[ablation] kernel=5 and kernel=9 training complete"
