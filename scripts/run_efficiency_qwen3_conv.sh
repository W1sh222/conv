#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B top-p efficiency benchmark: Full/Flex/XAttention/Conv/MInference.
# All user overrides belong after the .sh, for example:
#   bash scripts/run_efficiency_qwen3_conv.sh --threshold 0.9 --lengths 4,8,16,32,64,128
#   bash scripts/run_efficiency_qwen3_conv.sh --conv-weight-path /path/to/qwen.pt

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
cd "${REPO_ROOT}"

python -u eval/efficiency/attention_speedup_llama_conv.py \
  --model-kind qwen3 \
  --model-path "/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B" \
  --lengths "4,8,16,32,64,128" \
  --stride 8 \
  --threshold 0.9 \
  --conv-weight-path "/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn/qwen_weights/conv_qwen3_t065_64k128k_balanced_v3/stage3_extend_96k128k_t065_s8_yarn4_bf16_ema_step9250.pt" \
  --minference-vertical-size 512 \
  --minference-slash-size 3072 \
  --full-backend flashinfer \
  --capture-chunk-tokens 2048 \
  --method-chunk-size 32768 \
  --result-json "output/efficiency_qwen3_conv/results_minference_v512_s3072.json" \
  "$@"
