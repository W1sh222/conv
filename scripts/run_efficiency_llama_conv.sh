#!/usr/bin/env bash
set -euo pipefail

# Standalone upstream-style Llama top-p benchmark plus Conv.
# Existing scripts are intentionally untouched.
#
# Examples:
#   bash scripts/run_efficiency_llama_conv.sh
#   EFFICIENCY_LENGTHS=4,8,16,32,64,128 bash scripts/run_efficiency_llama_conv.sh
#   bash scripts/run_efficiency_llama_conv.sh --conv-weight-path /path/to/weight.pt

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
cd "${REPO_ROOT}"

python -u eval/efficiency/attention_speedup_llama_conv.py \
  --model-path "${LLAMA_MODEL_PATH:-/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct}" \
  --lengths "${EFFICIENCY_LENGTHS:-4,8,16,32,64,128}" \
  --stride "${STRIDE:-8}" \
  --threshold "${TOPP_THRESHOLD:-0.9}" \
  --conv-weight-path "${CONV_WEIGHT_PATH:-initial_vertical_diag}" \
  --full-backend "${FULL_BACKEND:-flashinfer}" \
  --capture-chunk-tokens "${CAPTURE_CHUNK_TOKENS:-2048}" \
  --method-chunk-size "${METHOD_CHUNK_SIZE:-32768}" \
  --result-json "${EFFICIENCY_RESULT_JSON:-output/efficiency_llama_conv/results.json}" \
  "$@"
