#!/usr/bin/env bash
set -euo pipefail

# LongBench-focused continuation at keep/top-k ratio 0.75.
#
# The 0.7 checkpoint is kept immutable.  This run starts from its EMA, moves
# the learned block budget to 0.75, and samples 8K-64K native-RoPE text.  The
# mix emphasizes QA/summarization-like RULER surrogates (qa_1/qa_2, vt, cwe,
# fwe, multiquery and multivalue) while retaining a small multikey replay so
# the RULER retrieval behavior does not collapse.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct}"
NOLIMA_ROOT="${NOLIMA_ROOT:-/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/dllm/data/NoLiMa}"
XATTN_ROOT="${XATTN_ROOT:-/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn}"

INIT_PATH="${INIT_PATH:-${XATTN_ROOT}/conv_weights/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_96k128k_continue_from4000_v1_bf16_ema.pt}"
RUN_NAME="${RUN_NAME:-conv_t075_longbench_from_t07_8k64k_v1}"
WEIGHT_DIR="${WEIGHT_DIR:-${XATTN_ROOT}/conv_weights/${RUN_NAME}}"
MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
DATA_SAMPLES="${DATA_SAMPLES:-12000}"
TRAIN_STEPS="${TRAIN_STEPS:-7000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-1}"
LR="${LR:-4e-7}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
AUTO_RESUME="${AUTO_RESUME:-1}"

THRESHOLD=0.75
BLOCK_TOPK_RATIO=0.75
MIN_SEQ_LENGTH=8192
MAX_SEQ_LENGTH=65536

DATA_DIR="${NOLIMA_ROOT}/synth_train/${RUN_NAME}"
SYNTH_DATA="${SYNTH_DATA:-${DATA_DIR}/ruler_mix_longbench_t075_8k64k_${DATA_SAMPLES}.jsonl}"
OUT_PATH="${OUT_PATH:-${WEIGHT_DIR}/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t075_longbench_from_t07_8k64k_${MODEL_PRECISION}.pt}"
STATE_PATH="${OUT_PATH%.pt}_train_state.pt"
EMA_PATH="${OUT_PATH%.pt}_ema.pt"
LOG_DIR="${LOG_DIR:-${WEIGHT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_longbench_t075_from_t07_${MODEL_PRECISION}.log}"

# Sum = 1.00.  QA and dense/general surrogates receive most of the new
# training budget; multikey replay is deliberately retained as an anchor.
TASK_MIX="${TASK_MIX:-niah_single_1:0.01,niah_single_2:0.01,niah_single_3:0.01,niah_multikey_1:0.08,niah_multikey_2:0.06,niah_multikey_3:0.03,niah_multivalue:0.10,niah_multiquery:0.12,vt:0.08,cwe:0.08,fwe:0.08,qa_1:0.18,qa_2:0.15,dense_general:0.01}"

mkdir -p "${WEIGHT_DIR}" "${DATA_DIR}" "${LOG_DIR}" "$(dirname "${OUT_PATH}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Working directory: $(pwd)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "INIT_PATH=${INIT_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "SYNTH_DATA=${SYNTH_DATA}"
echo "OUT_PATH=${OUT_PATH}"
echo "MODEL_PRECISION=${MODEL_PRECISION}"
echo "DATA_SAMPLES=${DATA_SAMPLES}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "LR=${LR}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "THRESHOLD=${THRESHOLD}"
echo "BLOCK_TOPK_RATIO=${BLOCK_TOPK_RATIO}"
echo "MIN_SEQ_LENGTH=${MIN_SEQ_LENGTH}"
echo "MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}"
echo "TASK_MIX=${TASK_MIX}"
echo "AUTO_RESUME=${AUTO_RESUME}"

test -d "${MODEL_PATH}" || { echo "model directory does not exist: ${MODEL_PATH}"; exit 1; }
test -d "${NOLIMA_ROOT}" || { echo "NoLiMa directory does not exist: ${NOLIMA_ROOT}"; exit 1; }
test -f "${INIT_PATH}" || { echo "initial EMA checkpoint does not exist: ${INIT_PATH}"; exit 1; }

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
    --min_seq_length "${MIN_SEQ_LENGTH}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --num_distractor_needles 64 \
    --position_mix "uniform:0.55,edge:0.25,bimodal:0.20" \
    --task_mix "${TASK_MIX}" \
    --seed 750864
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

if [[ -f "${EMA_PATH}" ]]; then
  echo "[train] completed checkpoint exists; skipping: ${EMA_PATH}"
  exit 0
fi

RESUME_ARGS=()
if [[ "${AUTO_RESUME}" == "1" && -f "${STATE_PATH}" ]]; then
  RESUME_ARGS+=(--resume_state "${STATE_PATH}")
  echo "[train] auto-resuming ${STATE_PATH}"
fi

python ft_scripts/sparse_ruler/train_conv_kernel_guarded_long.py \
  --model "${MODEL_PATH}" \
  --data "${SYNTH_DATA}" \
  --out "${OUT_PATH}" \
  --init_path "${INIT_PATH}" \
  "${RESUME_ARGS[@]}" \
  --model_type llama \
  --model_precision "${MODEL_PRECISION}" \
  --num_layers 32 \
  --num_heads 32 \
  --num_key_value_heads 8 \
  --kernel_size 7 \
  --layers_per_sample "${LAYERS_PER_SAMPLE}" \
  --min_seq_length "${MIN_SEQ_LENGTH}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --steps "${TRAIN_STEPS}" \
  --lr "${LR}" \
  --lr_schedule cosine \
  --warmup_steps "${WARMUP_STEPS}" \
  --min_lr_ratio 0.20 \
  --seed 750864 \
  --rope_scaling_type none \
  --rope_factor 4.0 \
  --rope_original_max_position_embeddings 32768 \
  --max_position_embeddings_override 131072 \
  --threshold "${THRESHOLD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO}" \
  --block_size 128 \
  --score_stride 8 \
  --score_chunk_size 0 \
  --score_sample_kernel_size 7 \
  --score_norm 1.0 \
  --teacher_rows 12 \
  --teacher_tail_rows 8 \
  --teacher_tokens_per_row 4 \
  --teacher_head_chunk 1 \
  --teacher_key_chunk 512 \
  --positive_temperature 0.015 \
  --teacher_kl_weight 0.55 \
  --teacher_l1_weight 0.08 \
  --topk_recall_loss_weight 0.60 \
  --topk_train_ratio 0.75 \
  --topk_positive_mass 0.99 \
  --topk_boundary_negatives 48 \
  --topk_boundary_margin 0.020 \
  --target_loss_weight 0.50 \
  --aggregation_loss_weight 0.12 \
  --aggregation_cover_mass 0.40 \
  --target_margin 0.08 \
  --target_joint_weight 0.80 \
  --target_query_tail_blocks 32 \
  --negative_weight 0.012 \
  --compression_loss_weight 0.0 \
  --compression_loss_weight_final 0.0 \
  --budget_loss_weight 0.0 \
  --target_blocks_schedule "0:384" \
  --bounded_delta_alpha 0.04 \
  --weight_min -1.0 \
  --weight_max 2.0 \
  --ema_decay 0.9997 \
  --max_grad_norm 0.03 \
  --log_steps 10 \
  --save_steps "${SAVE_STEPS}"

python ft_scripts/conv_ruler/verify_conv_kernel.py --path "${OUT_PATH}"
python ft_scripts/conv_ruler/verify_conv_kernel.py --path "${EMA_PATH}"

echo "LongBench-focused T0.75 continuation complete."
echo "Raw checkpoint: ${OUT_PATH}"
echo "EMA checkpoint: ${EMA_PATH}"
echo "Training state: ${STATE_PATH}"
