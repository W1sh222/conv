#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B counterpart of run_ruler_mix_sparse_guarded.sh.
# The data mix, 48K-64K length range, inference-matched scorer, sparse budget,
# losses, optimizer, and checkpoint cadence are kept identical to the Llama
# experiment. Qwen starts from an exact identity (no-convolution) 7x7 kernel;
# no Llama Conv checkpoint or optimizer state is reused.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

MODEL_PATH="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B"
NOLIMA_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/dllm/data/NoLiMa"
XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"
WEIGHT_DIR="${XATTN_ROOT}/conv_qwen3"

DATA_SAMPLES="${DATA_SAMPLES:-16000}"
# Keep a tokenizer-specific dataset: Qwen and Llama chat templates/token counts
# are different even though the synthetic task distribution is identical.
SYNTH_DATA="${SYNTH_DATA:-${NOLIMA_ROOT}/synth_train/qwen3_8b_ruler_mix_sparse_t065_48k64k_multikey_qa2_${DATA_SAMPLES}.jsonl}"

INIT_PATH="${INIT_PATH:-identity}"
MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
TRAIN_STEPS="${TRAIN_STEPS:-16000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-2}"
LR="${LR:-8e-6}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
WEIGHT_MIN="${WEIGHT_MIN:--1.0}"
WEIGHT_MAX="${WEIGHT_MAX:-2.0}"
THRESHOLD=0.65
BLOCK_TOPK_RATIO=0.65

OUT_PATH="${OUT_PATH:-${WEIGHT_DIR}/conv_kernel_7x7_qwen3_8b_ruler_mix_sparse_guarded_t065_multikey_qa2_48k64k_${MODEL_PRECISION}.pt}"
LOG_DIR="./ft_scripts/qwen_sparse_ruler/logs"
mkdir -p "${LOG_DIR}" "${WEIGHT_DIR}" "$(dirname "${SYNTH_DATA}")"
LOG_FILE="${LOG_DIR}/train_qwen3_8b_sparse_guarded_t065_multikey_qa2_48k64k_${MODEL_PRECISION}.log"
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
echo "LR_SCHEDULE=${LR_SCHEDULE}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "MIN_LR_RATIO=${MIN_LR_RATIO}"
echo "WEIGHT_MIN=${WEIGHT_MIN}"
echo "WEIGHT_MAX=${WEIGHT_MAX}"
echo "THRESHOLD=${THRESHOLD}"
echo "BLOCK_TOPK_RATIO=${BLOCK_TOPK_RATIO}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

test -d "${MODEL_PATH}" || { echo "model directory does not exist: ${MODEL_PATH}"; exit 1; }
test -d "${NOLIMA_ROOT}" || { echo "NoLiMa directory does not exist: ${NOLIMA_ROOT}"; exit 1; }
if [[ "${INIT_PATH}" != "identity" && "${INIT_PATH}" != "scratch_identity" && "${INIT_PATH}" != "no_conv" ]]; then
  test -f "${INIT_PATH}" || { echo "initial checkpoint does not exist: ${INIT_PATH}"; exit 1; }
fi

CURRENT_SAMPLES=0
if [[ -f "${SYNTH_DATA}" ]]; then
  CURRENT_SAMPLES="$(wc -l < "${SYNTH_DATA}")"
fi
if [[ "${REBUILD_DATA:-0}" == "1" || "${CURRENT_SAMPLES}" -ne "${DATA_SAMPLES}" ]]; then
  BUILD_PATH="${SYNTH_DATA}.building"
  echo "[data] rebuilding current=${CURRENT_SAMPLES} expected=${DATA_SAMPLES}"
  python ft_scripts/qwen_sparse_ruler/build_ruler_mix_sft.py \
    --model "${MODEL_PATH}" \
    --nolima_root "${NOLIMA_ROOT}" \
    --out "${BUILD_PATH}" \
    --haystack_subdirs rand_shuffle_long rand_shuffle \
    --num_samples "${DATA_SAMPLES}" \
    --min_seq_length 49152 \
    --max_seq_length 65536 \
    --num_distractor_needles 96 \
    --position_mix "uniform:0.65,edge:0.20,bimodal:0.15" \
    --task_mix "niah_single_1:0.005,niah_single_2:0.005,niah_single_3:0.005,niah_multikey_1:0.30,niah_multivalue:0.14,niah_multiquery:0.15,vt:0.05,cwe:0.08,fwe:0.04,qa_2:0.20,dense_general:0.025" \
    --seed 486465
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

RESUME_ARGS=()
if [[ -n "${RESUME_STATE:-}" ]]; then
  test -f "${RESUME_STATE}" || { echo "resume state does not exist: ${RESUME_STATE}"; exit 1; }
  RESUME_ARGS+=(--resume_state "${RESUME_STATE}")
fi

python ft_scripts/qwen_sparse_ruler/train_conv_kernel_guarded_long.py \
  --model "${MODEL_PATH}" \
  --data "${SYNTH_DATA}" \
  --out "${OUT_PATH}" \
  --init_path "${INIT_PATH}" \
  "${RESUME_ARGS[@]}" \
  --model_precision "${MODEL_PRECISION}" \
  --num_layers 36 \
  --num_heads 32 \
  --kernel_size 7 \
  --layers_per_sample "${LAYERS_PER_SAMPLE}" \
  --min_seq_length 49152 \
  --max_seq_length 65536 \
  --yarn_factor 4.0 \
  --steps "${TRAIN_STEPS}" \
  --lr "${LR}" \
  --lr_schedule "${LR_SCHEDULE}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --min_lr_ratio "${MIN_LR_RATIO}" \
  --threshold "${THRESHOLD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO}" \
  --block_size 128 \
  --score_stride 16 \
  --score_chunk_size 0 \
  --score_sample_kernel_size 7 \
  --score_norm 1.0 \
  --teacher_rows 16 \
  --teacher_tail_rows 12 \
  --teacher_tokens_per_row 4 \
  --teacher_head_chunk 1 \
  --teacher_key_chunk 512 \
  --positive_temperature 0.02 \
  --teacher_kl_weight 0.35 \
  --teacher_l1_weight 0.05 \
  --topk_recall_loss_weight 2.0 \
  --topk_train_ratio 0.65 \
  --topk_positive_mass 0.99 \
  --topk_boundary_negatives 32 \
  --topk_boundary_margin 0.03 \
  --target_loss_weight 1.20 \
  --aggregation_loss_weight 0.25 \
  --aggregation_cover_mass 0.45 \
  --target_margin 0.12 \
  --target_joint_weight 0.90 \
  --target_query_tail_blocks 64 \
  --negative_weight 0.03 \
  --compression_loss_weight 0.0 \
  --compression_loss_weight_final 0.0 \
  --budget_loss_weight 0.0 \
  --target_blocks_schedule "0:256" \
  --weight_min "${WEIGHT_MIN}" \
  --weight_max "${WEIGHT_MAX}" \
  --ema_decay 0.997 \
  --max_grad_norm 0.05 \
  --log_steps 10 \
  --save_steps "${SAVE_STEPS}"

python ft_scripts/qwen_sparse_ruler/verify_qwen_conv_kernel.py --path "${OUT_PATH}"
python ft_scripts/qwen_sparse_ruler/verify_qwen_conv_kernel.py --path "${OUT_PATH%.pt}_ema.pt"

echo "Qwen3-8B 48K-64K T0.65 multikey/QA2 training complete."
echo "Raw checkpoint: ${OUT_PATH}"
echo "EMA checkpoint: ${OUT_PATH%.pt}_ema.pt"
echo "Training state: ${OUT_PATH%.pt}_train_state.pt"
