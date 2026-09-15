#!/usr/bin/env bash
set -euo pipefail

# Short RULER-focused continuation from the known-good Llama Stage-3 EMA.
# This is a new run: optimizer state is reset unless RESUME_STATE is explicitly
# supplied.  Stage-3 and previous corrective checkpoints are never overwritten.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct"
NOLIMA_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/dllm/data/NoLiMa"
XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"

# INIT is weights-only; do not import the failed Stage-4 optimizer state.
INIT_PATH="${INIT_PATH:-${XATTN_ROOT}/llama_weights/conv_llama_t065_scratch_curriculum_v1/stage3_48k64k_t065_bf16_ema.pt}"
RUN_NAME="${RUN_NAME:-conv_llama_t065_stage3_shortmix_32k48k_v1}"
WEIGHT_DIR="${XATTN_ROOT}/llama_weights/${RUN_NAME}"
DATA_DIR="${NOLIMA_ROOT}/synth_train/${RUN_NAME}"
LOG_DIR="${WEIGHT_DIR}/logs"

MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
DATA_SAMPLES="${DATA_SAMPLES:-16000}"
TRAIN_STEPS="${TRAIN_STEPS:-3000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-4}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
THRESHOLD="${THRESHOLD:-0.65}"
TOPK_RATIO="${TOPK_RATIO:-0.65}"
SCORE_STRIDE="${SCORE_STRIDE:-8}"

DATA_PATH="${DATA_DIR}/ruler_mix_32k48k_${DATA_SAMPLES}.jsonl"
OUT_PATH="${WEIGHT_DIR}/conv_kernel_7x7_llama_t065_stage3_shortmix_32k48k_${MODEL_PRECISION}.pt"
STATE_PATH="${OUT_PATH%.pt}_train_state.pt"
LOG_FILE="${LOG_DIR}/train_llama_t065_stage3_shortmix_${MODEL_PRECISION}.log"

# Reuse the task distribution from the old strong short corrective run.
TASK_MIX="niah_single_1:0.01,niah_single_2:0.01,niah_single_3:0.01,niah_multikey_1:0.20,niah_multivalue:0.20,niah_multiquery:0.23,vt:0.07,cwe:0.11,fwe:0.05,qa_2:0.09,dense_general:0.02"

mkdir -p "${WEIGHT_DIR}" "${DATA_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Working directory: $(pwd)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "INIT_PATH=${INIT_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "DATA_PATH=${DATA_PATH}"
echo "OUT_PATH=${OUT_PATH}"
echo "MODEL_PRECISION=${MODEL_PRECISION}"
echo "DATA_SAMPLES=${DATA_SAMPLES}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "LAYERS_PER_SAMPLE=${LAYERS_PER_SAMPLE}"
echo "LR=${LR}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "LENGTH=32768-49152"
echo "TOPK_RATIO=${TOPK_RATIO}"
echo "BLOCK_SIZE=128"
echo "SCORE_STRIDE=${SCORE_STRIDE}"
echo "TASK_MIX=${TASK_MIX}"

test -d "${MODEL_PATH}" || { echo "model directory does not exist: ${MODEL_PATH}"; exit 1; }
test -d "${NOLIMA_ROOT}" || { echo "NoLiMa directory does not exist: ${NOLIMA_ROOT}"; exit 1; }
test -f "${INIT_PATH}" || { echo "INIT checkpoint does not exist: ${INIT_PATH}"; exit 1; }

# Rebuild atomically when requested or if the cached line count is incorrect.
CURRENT_SAMPLES=0
if [[ -f "${DATA_PATH}" ]]; then
  CURRENT_SAMPLES="$(wc -l < "${DATA_PATH}")"
fi
if [[ "${REBUILD_DATA:-0}" == "1" || "${CURRENT_SAMPLES}" -ne "${DATA_SAMPLES}" ]]; then
  BUILD_PATH="${DATA_PATH}.building"
  echo "[data] rebuilding current=${CURRENT_SAMPLES} expected=${DATA_SAMPLES}"
  python ft_scripts/sparse_ruler/build_ruler_mix_sft.py \
    --model "${MODEL_PATH}" \
    --nolima_root "${NOLIMA_ROOT}" \
    --out "${BUILD_PATH}" \
    --haystack_subdirs rand_shuffle_long rand_shuffle \
    --num_samples "${DATA_SAMPLES}" \
    --min_seq_length 32768 \
    --max_seq_length 49152 \
    --num_distractor_needles 72 \
    --position_mix "uniform:0.60,edge:0.25,bimodal:0.15" \
    --task_mix "${TASK_MIX}" \
    --seed 325048
  BUILT_SAMPLES="$(wc -l < "${BUILD_PATH}")"
  test "${BUILT_SAMPLES}" -eq "${DATA_SAMPLES}" || {
    echo "incomplete dataset: ${BUILT_SAMPLES}/${DATA_SAMPLES}"
    exit 1
  }
  mv -f "${BUILD_PATH}" "${DATA_PATH}"
fi

CURRENT_SAMPLES="$(wc -l < "${DATA_PATH}")"
test "${CURRENT_SAMPLES}" -eq "${DATA_SAMPLES}" || {
  echo "dataset integrity failure: ${CURRENT_SAMPLES}/${DATA_SAMPLES}"
  exit 1
}

# Fresh optimizer by default; explicitly set RESUME_STATE only to recover this
# exact run. A Stage-4 or other experiment state must not be passed here.
RESUME_ARGS=()
if [[ -n "${RESUME_STATE:-}" ]]; then
  test -f "${RESUME_STATE}" || { echo "resume state does not exist: ${RESUME_STATE}"; exit 1; }
  RESUME_ARGS+=(--resume_state "${RESUME_STATE}")
fi

python ft_scripts/sparse_ruler/train_conv_kernel_guarded_long.py \
  --model "${MODEL_PATH}" \
  --data "${DATA_PATH}" \
  --out "${OUT_PATH}" \
  --init_path "${INIT_PATH}" \
  "${RESUME_ARGS[@]}" \
  --model_precision "${MODEL_PRECISION}" \
  --model_type llama \
  --num_layers 32 \
  --num_heads 32 \
  --num_key_value_heads 8 \
  --kernel_size 7 \
  --layers_per_sample "${LAYERS_PER_SAMPLE}" \
  --min_seq_length 32768 \
  --max_seq_length 49152 \
  --steps "${TRAIN_STEPS}" \
  --lr "${LR}" \
  --lr_schedule cosine \
  --warmup_steps "${WARMUP_STEPS}" \
  --min_lr_ratio "${MIN_LR_RATIO}" \
  --seed 325049 \
  --threshold "${THRESHOLD}" \
  --block_topk_ratio "${TOPK_RATIO}" \
  --block_size 128 \
  --score_stride "${SCORE_STRIDE}" \
  --score_chunk_size 0 \
  --score_sample_kernel_size 7 \
  --score_norm 1.0 \
  --teacher_rows 16 \
  --teacher_tail_rows 12 \
  --teacher_tokens_per_row 4 \
  --teacher_head_chunk 2 \
  --teacher_key_chunk 1024 \
  --positive_temperature 0.02 \
  --teacher_kl_weight 0.25 \
  --teacher_l1_weight 0.05 \
  --topk_recall_loss_weight 5.0 \
  --topk_train_ratio "${TOPK_RATIO}" \
  --topk_positive_mass 1.0 \
  --topk_boundary_negatives 32 \
  --topk_boundary_margin 0.03 \
  --target_loss_weight 0.80 \
  --aggregation_loss_weight 0.30 \
  --aggregation_cover_mass 0.45 \
  --target_margin 0.12 \
  --target_joint_weight 0.90 \
  --target_query_tail_blocks 32 \
  --negative_weight 0.05 \
  --compression_loss_weight 0.0 \
  --compression_loss_weight_final 0.0 \
  --budget_loss_weight 0.0 \
  --target_blocks_schedule "0:256" \
  --weight_min -1.0 \
  --weight_max 2.0 \
  --ema_decay 0.997 \
  --max_grad_norm 0.05 \
  --log_steps 10 \
  --save_steps "${SAVE_STEPS}"

python ft_scripts/sparse_ruler/verify_conv_kernel.py \
  --path "${OUT_PATH}" --num_layers 32 --num_heads 32 --kernel_size 7
python ft_scripts/sparse_ruler/verify_conv_kernel.py \
  --path "${OUT_PATH%.pt}_ema.pt" --num_layers 32 --num_heads 32 --kernel_size 7

echo "Short Llama T0.65 RULER continuation complete."
echo "Raw checkpoint: ${OUT_PATH}"
echo "EMA checkpoint: ${OUT_PATH%.pt}_ema.pt"
echo "Training state: ${STATE_PATH}"
echo "Step checkpoints: ${OUT_PATH%.pt}_ema_step<N>.pt"
