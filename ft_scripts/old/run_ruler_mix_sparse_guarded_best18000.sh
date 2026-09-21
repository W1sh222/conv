#!/usr/bin/env bash
set -euo pipefail

# Long-context quality training for the original sparse Conv architecture.
# Context: 24K-32K. Retrieval surface: 70% of causal-visible key blocks.
# Frozen Llama: BF16 by default; trainable 7x7 Conv and teacher math: FP32.
#
# This is a new experiment and never overwrites the existing T0.8 or the
# plateaued T0.8 24K-32K checkpoints.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

MODEL_PATH="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct"
NOLIMA_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/dllm/data/NoLiMa"
XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"
WEIGHT_DIR="${XATTN_ROOT}/conv_weights"

DATA_SAMPLES="${DATA_SAMPLES:-12000}"
# The synthetic data is independent of selector threshold, so reuse the
# already-built, integrity-checked 24K-32K dataset when present.
SYNTH_DATA="${SYNTH_DATA:-${NOLIMA_ROOT}/synth_train/ruler_mix_sparse_t08_24k32k_${DATA_SAMPLES}.jsonl}"

# Start at the exact no-convolution baseline. This avoids inheriting the
# previously underperforming learned convolution while preserving a safe score
# map at step zero. Set INIT_PATH to a real checkpoint only when desired.
INIT_PATH="${INIT_PATH:-identity}"

MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
TRAIN_STEPS="${TRAIN_STEPS:-18000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-4}"
LR="${LR:-2e-5}"
THRESHOLD=0.7
BLOCK_TOPK_RATIO=0.7

OUT_PATH="${OUT_PATH:-${WEIGHT_DIR}/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_24k32k_${MODEL_PRECISION}.pt}"
LOG_DIR="./ft_scripts/sparse_ruler/logs"
mkdir -p "${LOG_DIR}" "${WEIGHT_DIR}" "$(dirname "${SYNTH_DATA}")"
LOG_FILE="${LOG_DIR}/train_sparse_guarded_long_t07_24k32k_${MODEL_PRECISION}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Working directory: $(pwd)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "NOLIMA_ROOT=${NOLIMA_ROOT}"
echo "SYNTH_DATA=${SYNTH_DATA}"
echo "DATA_SAMPLES=${DATA_SAMPLES}"
echo "INIT_PATH=${INIT_PATH}"
echo "OUT_PATH=${OUT_PATH}"
echo "MODEL_PRECISION=${MODEL_PRECISION}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "LAYERS_PER_SAMPLE=${LAYERS_PER_SAMPLE}"
echo "LR=${LR}"
echo "THRESHOLD=${THRESHOLD}"
echo "BLOCK_TOPK_RATIO=${BLOCK_TOPK_RATIO}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

test -d "${MODEL_PATH}" || { echo "model directory does not exist: ${MODEL_PATH}"; exit 1; }
test -d "${NOLIMA_ROOT}" || { echo "NoLiMa directory does not exist: ${NOLIMA_ROOT}"; exit 1; }
if [[ "${INIT_PATH}" != "identity" && "${INIT_PATH}" != "scratch_identity" && "${INIT_PATH}" != "no_conv" ]]; then
  test -f "${INIT_PATH}" || { echo "initial checkpoint does not exist: ${INIT_PATH}"; exit 1; }
fi

# Build to a temporary file, verify the exact line count, then publish it.
# An interrupted builder can never leave a partial dataset that is reused.
CURRENT_SAMPLES=0
if [[ -f "${SYNTH_DATA}" ]]; then
  CURRENT_SAMPLES="$(wc -l < "${SYNTH_DATA}")"
fi
if [[ "${REBUILD_DATA:-0}" == "1" || "${CURRENT_SAMPLES}" -ne "${DATA_SAMPLES}" ]]; then
  BUILD_PATH="${SYNTH_DATA}.building"
  echo "[data] rebuilding current=${CURRENT_SAMPLES} expected=${DATA_SAMPLES}"
  python ft_scripts/sparse_ruler/build_ruler_mix_sft.py \
    --model "${MODEL_PATH}" \
    --nolima_root "${NOLIMA_ROOT}" \
    --out "${BUILD_PATH}" \
    --haystack_subdirs rand_shuffle_long rand_shuffle \
    --num_samples "${DATA_SAMPLES}" \
    --min_seq_length 24576 \
    --max_seq_length 32768 \
    --num_distractor_needles 64 \
    --position_mix "uniform:0.60,edge:0.25,bimodal:0.15" \
    --task_mix "niah_single_1:0.01,niah_single_2:0.01,niah_single_3:0.01,niah_multikey_1:0.18,niah_multivalue:0.20,niah_multiquery:0.22,vt:0.08,cwe:0.10,fwe:0.06,qa_2:0.115,dense_general:0.015" \
    --seed 24032
  BUILT_SAMPLES="$(wc -l < "${BUILD_PATH}")"
  test "${BUILT_SAMPLES}" -eq "${DATA_SAMPLES}" || {
    echo "incomplete dataset: ${BUILT_SAMPLES}/${DATA_SAMPLES}"
    exit 1
  }
  mv -f "${BUILD_PATH}" "${SYNTH_DATA}"
fi

CURRENT_SAMPLES="$(wc -l < "${SYNTH_DATA}")"
test "${CURRENT_SAMPLES}" -eq "${DATA_SAMPLES}" || {
  echo "dataset integrity failure: ${CURRENT_SAMPLES}/${DATA_SAMPLES}"
  exit 1
}

# RESUME_STATE is only for resuming this exact T0.7 experiment. Do not pass a
# T0.8 optimizer state. INIT_PATH is ignored after a resume state is restored.
RESUME_ARGS=()
if [[ -n "${RESUME_STATE:-}" ]]; then
  test -f "${RESUME_STATE}" || { echo "resume state does not exist: ${RESUME_STATE}"; exit 1; }
  RESUME_ARGS+=(--resume_state "${RESUME_STATE}")
fi

python ft_scripts/sparse_ruler/train_conv_kernel_guarded_long.py \
  --model "${MODEL_PATH}" \
  --data "${SYNTH_DATA}" \
  --out "${OUT_PATH}" \
  --init_path "${INIT_PATH}" \
  "${RESUME_ARGS[@]}" \
  --model_precision "${MODEL_PRECISION}" \
  --num_layers 32 \
  --num_heads 32 \
  --kernel_size 7 \
  --layers_per_sample "${LAYERS_PER_SAMPLE}" \
  --min_seq_length 24576 \
  --max_seq_length 32768 \
  --steps "${TRAIN_STEPS}" \
  --lr "${LR}" \
  --threshold "${THRESHOLD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO}" \
  --block_size 128 \
  --score_stride 16 \
  --score_chunk_size 0 \
  --score_sample_kernel_size 7 \
  --score_norm 1.0 \
  --teacher_rows 12 \
  --teacher_tail_rows 8 \
  --teacher_tokens_per_row 4 \
  --teacher_head_chunk 2 \
  --teacher_key_chunk 1024 \
  --positive_temperature 0.01 \
  --teacher_kl_weight 0.40 \
  --teacher_l1_weight 0.10 \
  --target_loss_weight 0.55 \
  --aggregation_loss_weight 0.18 \
  --aggregation_cover_mass 0.32 \
  --target_margin 0.10 \
  --target_joint_weight 0.90 \
  --target_query_tail_blocks 24 \
  --negative_weight 0.02 \
  --compression_loss_weight 0.0 \
  --compression_loss_weight_final 0.0 \
  --budget_loss_weight 0.0 \
  --target_blocks_schedule "0:256" \
  --bounded_delta_alpha 0.08 \
  --ema_decay 0.997 \
  --max_grad_norm 0.05 \
  --log_steps 10 \
  --save_steps "${SAVE_STEPS}"

python ft_scripts/conv_ruler/verify_conv_kernel.py --path "${OUT_PATH}"
python ft_scripts/conv_ruler/verify_conv_kernel.py --path "${OUT_PATH%.pt}_ema.pt"

echo "Long-context T0.7 quality training complete."
echo "Raw checkpoint: ${OUT_PATH}"
echo "EMA checkpoint: ${OUT_PATH%.pt}_ema.pt"
echo "Training state: ${OUT_PATH%.pt}_train_state.pt"
