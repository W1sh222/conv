#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B corrective Stage 4.
#
# This is intentionally a separate continuation entry point.  Stages 1-3 of
# conv_qwen3_t065_64k128k_balanced_v3 are left untouched; the input is the
# last requested Stage-3 EMA (step 9250), and this stage performs a cautious
# native-RoPE 8K-64K correction for LongBench/short-context regressions.
# All score maps remain inference-matched: block_size=128 and score_stride=8.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B"
NOLIMA_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/dllm/data/NoLiMa"
XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"

# Keep the original Stage-3 result as the immutable starting point.
STAGE3_INIT="${STAGE3_INIT:-${XATTN_ROOT}/qwen_weights/conv_qwen3_t065_64k128k_balanced_v3/stage3_extend_96k128k_t065_s8_yarn4_bf16_ema_step9250.pt}"
RUN_NAME="${RUN_NAME:-conv_qwen3_t065_64k128k_compensate_stage4_v1}"
WEIGHT_DIR="${XATTN_ROOT}/qwen_weights/${RUN_NAME}"
DATA_DIR="${NOLIMA_ROOT}/synth_train/${RUN_NAME}"
LOG_DIR="${WEIGHT_DIR}/logs"

MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
DATA_SAMPLES="${DATA_SAMPLES:-8000}"
TRAIN_STEPS="${TRAIN_STEPS:-8000}"
LR="${LR:-7e-7}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-2}"
SAVE_STEPS="${SAVE_STEPS:-250}"
AUTO_RESUME="${AUTO_RESUME:-1}"

DATA_PATH="${DATA_DIR}/stage4_compensate_native_8k64k_${DATA_SAMPLES}.jsonl"
OUT_PATH="${WEIGHT_DIR}/conv_kernel_7x7_qwen3_t065_compensate_native_8k64k_s8_${MODEL_PRECISION}.pt"
STATE_PATH="${OUT_PATH%.pt}_train_state.pt"
EMA_PATH="${OUT_PATH%.pt}_ema.pt"
LOG_FILE="${LOG_DIR}/train_qwen3_t065_compensate_stage4_${MODEL_PRECISION}.log"

# More QA/dense data repairs the LongBench regression while retaining enough
# multikey/multivalue data to avoid throwing away the RULER behavior learned in
# the Stage-3 anchor.
TASK_MIX="niah_single_1:0.005,niah_single_2:0.005,niah_single_3:0.005,niah_multikey_1:0.15,niah_multivalue:0.10,niah_multiquery:0.10,vt:0.04,cwe:0.05,fwe:0.03,qa_1:0.15,qa_2:0.15,dense_general:0.215"

mkdir -p "${WEIGHT_DIR}" "${DATA_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Working directory: $(pwd)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "STAGE3_INIT=${STAGE3_INIT}"
echo "RUN_NAME=${RUN_NAME}"
echo "DATA_PATH=${DATA_PATH}"
echo "OUT_PATH=${OUT_PATH}"
echo "MODEL_PRECISION=${MODEL_PRECISION}"
echo "DATA_SAMPLES=${DATA_SAMPLES}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "LR=${LR}"
echo "LAYERS_PER_SAMPLE=${LAYERS_PER_SAMPLE}"
echo "BLOCK_SIZE=128"
echo "SCORE_STRIDE=8"
echo "ROPE_SCALING_TYPE=none"
echo "TASK_MIX=${TASK_MIX}"
echo "AUTO_RESUME=${AUTO_RESUME}"

test -d "${MODEL_PATH}" || { echo "model directory does not exist: ${MODEL_PATH}"; exit 1; }
test -d "${NOLIMA_ROOT}" || { echo "NoLiMa directory does not exist: ${NOLIMA_ROOT}"; exit 1; }
test -f "${STAGE3_INIT}" || { echo "Stage-3 anchor does not exist: ${STAGE3_INIT}"; exit 1; }

CURRENT_SAMPLES=0
if [[ -f "${DATA_PATH}" ]]; then
  CURRENT_SAMPLES="$(wc -l < "${DATA_PATH}")"
fi
if [[ "${REBUILD_DATA:-0}" == "1" || "${CURRENT_SAMPLES}" -ne "${DATA_SAMPLES}" ]]; then
  BUILD_PATH="${DATA_PATH}.building"
  echo "[data] rebuilding current=${CURRENT_SAMPLES} expected=${DATA_SAMPLES}"
  python ft_scripts/sparse_ruler_qwen/build_ruler_mix_sft.py \
    --model "${MODEL_PATH}" \
    --nolima_root "${NOLIMA_ROOT}" \
    --out "${BUILD_PATH}" \
    --haystack_subdirs rand_shuffle_long rand_shuffle \
    --num_samples "${DATA_SAMPLES}" \
    --min_seq_length 8192 \
    --max_seq_length 65536 \
    --num_distractor_needles 80 \
    --position_mix "uniform:0.55,edge:0.25,bimodal:0.20" \
    --task_mix "${TASK_MIX}" \
    --seed 651604
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

if [[ -f "${EMA_PATH}" ]]; then
  echo "[stage4] completed checkpoint exists; skipping: ${EMA_PATH}"
  exit 0
fi

RESUME_ARGS=()
if [[ "${AUTO_RESUME}" == "1" && -f "${STATE_PATH}" ]]; then
  RESUME_ARGS+=(--resume_state "${STATE_PATH}")
  echo "[stage4] auto-resuming ${STATE_PATH}"
fi

python ft_scripts/sparse_ruler_qwen/train_conv_kernel_guarded_long.py \
  --model "${MODEL_PATH}" \
  --data "${DATA_PATH}" \
  --out "${OUT_PATH}" \
  --init_path "${STAGE3_INIT}" \
  "${RESUME_ARGS[@]}" \
  --model_precision "${MODEL_PRECISION}" \
  --model_type qwen3 \
  --num_layers 36 \
  --num_heads 32 \
  --num_key_value_heads 8 \
  --kernel_size 7 \
  --layers_per_sample "${LAYERS_PER_SAMPLE}" \
  --min_seq_length 8192 \
  --max_seq_length 65536 \
  --steps "${TRAIN_STEPS}" \
  --lr "${LR}" \
  --lr_schedule cosine \
  --warmup_steps "${WARMUP_STEPS}" \
  --min_lr_ratio 0.10 \
  --seed 651604 \
  --rope_scaling_type none \
  --rope_factor 4.0 \
  --rope_original_max_position_embeddings 32768 \
  --max_position_embeddings_override 0 \
  --threshold 0.65 \
  --block_topk_ratio 0.65 \
  --block_size 128 \
  --score_stride 8 \
  --score_chunk_size 0 \
  --score_sample_kernel_size 7 \
  --score_norm 1.0 \
  --teacher_rows 20 \
  --teacher_tail_rows 12 \
  --teacher_tokens_per_row 4 \
  --teacher_head_chunk 1 \
  --teacher_key_chunk 512 \
  --positive_temperature 0.02 \
  --teacher_kl_weight 0.50 \
  --teacher_l1_weight 0.08 \
  --topk_recall_loss_weight 1.00 \
  --topk_train_ratio 0.65 \
  --topk_positive_mass 0.99 \
  --topk_boundary_negatives 32 \
  --topk_boundary_margin 0.03 \
  --target_loss_weight 0.65 \
  --aggregation_loss_weight 0.18 \
  --aggregation_cover_mass 0.45 \
  --target_margin 0.10 \
  --target_joint_weight 0.80 \
  --target_query_tail_blocks 32 \
  --negative_weight 0.02 \
  --compression_loss_weight 0.0 \
  --compression_loss_weight_final 0.0 \
  --budget_loss_weight 0.0 \
  --target_blocks_schedule "0:512" \
  --bounded_delta_alpha 0.04 \
  --ema_decay 0.999 \
  --max_grad_norm 0.05 \
  --log_steps 10 \
  --save_steps "${SAVE_STEPS}"

python ft_scripts/sparse_ruler_qwen/verify_qwen_conv_kernel.py --path "${OUT_PATH}"
python ft_scripts/sparse_ruler_qwen/verify_qwen_conv_kernel.py --path "${EMA_PATH}"

echo "Qwen corrective Stage 4 complete."
echo "Raw checkpoint: ${OUT_PATH}"
echo "EMA checkpoint: ${EMA_PATH}"
echo "Training state: ${STATE_PATH}"
