#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RULER 24K-32K T0.7 hard continuation
#
# Base:
#   conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_24k32k_bf16_step18000.pt
#
# Goal:
#   Fine-tune the already strong step18000 checkpoint on harder retrieval tasks.
#
# IMPORTANT:
#   - Fresh optimizer
#   - DO NOT use old RESUME_STATE
#   - DO NOT use absolute weight range / expand parameterization
#   - Keep topk_ratio = 0.7 exactly
#   - Use step18000 as residual anchor
#
# Current trainer DOES NOT support:
#   --lr_schedule
#   --warmup_steps
#   --min_lr_ratio
#
# Therefore this script uses a conservative constant LR = 3e-6.
# ==============================================================================


# ==============================================================================
# Environment
# ==============================================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"


# ==============================================================================
# Paths
# ==============================================================================

MODEL_PATH="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct"

NOLIMA_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/dllm/data/NoLiMa"

XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"

WEIGHT_DIR="${XATTN_ROOT}/conv_weights"


# ==============================================================================
# Anchor checkpoint
# ==============================================================================

INIT_PATH="${WEIGHT_DIR}/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_24k32k_bf16_step18000.pt"


# ==============================================================================
# Hard continuation dataset
#
# Your previous attempt already generated this successfully.
# It will be reused automatically when it contains 12000 lines.
# ==============================================================================

DATA_SAMPLES="${DATA_SAMPLES:-12000}"

SYNTH_DATA="${SYNTH_DATA:-${NOLIMA_ROOT}/synth_train/ruler_mix_sparse_t07_hardcont_24k32k_${DATA_SAMPLES}.jsonl}"


# ==============================================================================
# Training parameters
# ==============================================================================

MODEL_PRECISION="${MODEL_PRECISION:-bf16}"

# 8000 is the maximum continuation horizon.
# Do not assume the final step is the best checkpoint.
TRAIN_STEPS="${TRAIN_STEPS:-8000}"

SAVE_STEPS="${SAVE_STEPS:-250}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-4}"

# Current trainer has constant LR only.
#
# The previous proposed 5e-6 cosine run would have had an average LR much lower
# than 5e-6, so 3e-6 constant is a safer replacement than 5e-6 constant.
LR="${LR:-3e-6}"

SEED="${SEED:-24033}"

THRESHOLD="${THRESHOLD:-0.7}"
BLOCK_TOPK_RATIO="${BLOCK_TOPK_RATIO:-0.7}"

# New checkpoint is the anchor:
#
#   W = W_step18000 + 0.035 * tanh(delta)
#
# This prevents the expand-style unrestricted drift.
BOUNDED_DELTA_ALPHA="${BOUNDED_DELTA_ALPHA:-0.035}"


# ==============================================================================
# Output
# ==============================================================================

OUT_PATH="${OUT_PATH:-${WEIGHT_DIR}/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_step18000_hardcont_lr3e-6_a0035_24k32k_bf16.pt}"


# ==============================================================================
# Logging
# ==============================================================================

LOG_DIR="./ft_scripts/sparse_ruler/logs"

mkdir -p \
    "${LOG_DIR}" \
    "${WEIGHT_DIR}" \
    "$(dirname "${SYNTH_DATA}")"

LOG_FILE="${LOG_DIR}/train_t07_step18000_hardcont_lr3e-6_a0035_24k32k_bf16.log"

exec > >(tee -a "${LOG_FILE}") 2>&1


echo "=============================================================================="
echo "RULER long-context T0.7 hard continuation"
echo "=============================================================================="

echo "Working directory      : $(pwd)"
echo "CUDA_VISIBLE_DEVICES   : ${CUDA_VISIBLE_DEVICES}"
echo

echo "MODEL_PATH             : ${MODEL_PATH}"
echo "NOLIMA_ROOT            : ${NOLIMA_ROOT}"
echo

echo "INIT_PATH              : ${INIT_PATH}"
echo "SYNTH_DATA             : ${SYNTH_DATA}"
echo "OUT_PATH               : ${OUT_PATH}"
echo

echo "DATA_SAMPLES           : ${DATA_SAMPLES}"
echo "MODEL_PRECISION        : ${MODEL_PRECISION}"
echo "TRAIN_STEPS            : ${TRAIN_STEPS}"
echo "SAVE_STEPS             : ${SAVE_STEPS}"
echo "LAYERS_PER_SAMPLE      : ${LAYERS_PER_SAMPLE}"
echo

echo "LR                      : ${LR}"
echo "THRESHOLD               : ${THRESHOLD}"
echo "BLOCK_TOPK_RATIO        : ${BLOCK_TOPK_RATIO}"
echo "BOUNDED_DELTA_ALPHA     : ${BOUNDED_DELTA_ALPHA}"
echo "SEED                    : ${SEED}"
echo

echo "PYTORCH_CUDA_ALLOC_CONF : ${PYTORCH_CUDA_ALLOC_CONF}"

echo "=============================================================================="


# ==============================================================================
# Sanity checks
# ==============================================================================

test -d "${MODEL_PATH}" || {
    echo "ERROR: model directory does not exist:"
    echo "${MODEL_PATH}"
    exit 1
}

test -d "${NOLIMA_ROOT}" || {
    echo "ERROR: NoLiMa directory does not exist:"
    echo "${NOLIMA_ROOT}"
    exit 1
}

test -f "${INIT_PATH}" || {
    echo "ERROR: anchor checkpoint does not exist:"
    echo "${INIT_PATH}"
    exit 1
}


# ==============================================================================
# 0. Verify anchor
# ==============================================================================

echo
echo "=============================================================================="
echo "[0/3] Verify step18000 anchor"
echo "=============================================================================="

python ft_scripts/conv_ruler/verify_conv_kernel.py \
    --path "${INIT_PATH}"


# ==============================================================================
# 1. Build / reuse hard continuation dataset
# ==============================================================================

CURRENT_SAMPLES=0

if [[ -f "${SYNTH_DATA}" ]]; then
    CURRENT_SAMPLES="$(wc -l < "${SYNTH_DATA}")"
fi


if [[ "${REBUILD_DATA:-0}" == "1" || "${CURRENT_SAMPLES}" -ne "${DATA_SAMPLES}" ]]; then

    BUILD_PATH="${SYNTH_DATA}.building"

    rm -f "${BUILD_PATH}"

    echo
    echo "=============================================================================="
    echo "[1/3] Build hard continuation dataset"
    echo "=============================================================================="

    echo "Existing samples : ${CURRENT_SAMPLES}"
    echo "Expected samples : ${DATA_SAMPLES}"


    python ft_scripts/sparse_ruler/build_ruler_mix_sft.py \
        --model "${MODEL_PATH}" \
        --nolima_root "${NOLIMA_ROOT}" \
        --out "${BUILD_PATH}" \
        \
        --haystack_subdirs rand_shuffle_long rand_shuffle \
        \
        --num_samples "${DATA_SAMPLES}" \
        \
        --min_seq_length 24576 \
        --max_seq_length 32768 \
        \
        --num_distractor_needles 64 \
        \
        --position_mix "uniform:0.60,edge:0.25,bimodal:0.15" \
        \
        --task_mix "niah_single_1:0.005,niah_single_2:0.005,niah_single_3:0.005,niah_multikey_1:0.10,niah_multikey_2:0.18,niah_multikey_3:0.08,niah_multivalue:0.12,niah_multiquery:0.11,vt:0.09,cwe:0.04,fwe:0.04,qa_1:0.13,qa_2:0.08,dense_general:0.015" \
        \
        --seed "${SEED}"


    BUILT_SAMPLES="$(wc -l < "${BUILD_PATH}")"

    if [[ "${BUILT_SAMPLES}" -ne "${DATA_SAMPLES}" ]]; then

        echo "ERROR: incomplete dataset"
        echo "built=${BUILT_SAMPLES}"
        echo "expected=${DATA_SAMPLES}"

        exit 1
    fi


    mv -f "${BUILD_PATH}" "${SYNTH_DATA}"

else

    echo
    echo "=============================================================================="
    echo "[1/3] Reuse existing hard continuation dataset"
    echo "=============================================================================="

    echo "dataset=${SYNTH_DATA}"
    echo "samples=${CURRENT_SAMPLES}"

fi


# Final integrity check

CURRENT_SAMPLES="$(wc -l < "${SYNTH_DATA}")"

if [[ "${CURRENT_SAMPLES}" -ne "${DATA_SAMPLES}" ]]; then

    echo "ERROR: dataset integrity failure"
    echo "actual=${CURRENT_SAMPLES}"
    echo "expected=${DATA_SAMPLES}"

    exit 1
fi


# ==============================================================================
# 2. Continuation training
#
# Current trainer supported parameters only.
#
# NOTE:
#   There is intentionally NO:
#
#       --resume_state
#       --lr_schedule
#       --warmup_steps
#       --min_lr_ratio
#       --weight_min
#       --weight_max
#
# ==============================================================================

echo
echo "=============================================================================="
echo "[2/3] Start continuation training"
echo "=============================================================================="

python ft_scripts/sparse_ruler/train_conv_kernel_guarded_long.py \
    \
    --model "${MODEL_PATH}" \
    --data "${SYNTH_DATA}" \
    --out "${OUT_PATH}" \
    --init_path "${INIT_PATH}" \
    \
    --model_precision "${MODEL_PRECISION}" \
    \
    --num_layers 32 \
    --num_heads 32 \
    --kernel_size 7 \
    \
    --layers_per_sample "${LAYERS_PER_SAMPLE}" \
    \
    --min_seq_length 24576 \
    --max_seq_length 32768 \
    \
    --steps "${TRAIN_STEPS}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    \
    --block_size 128 \
    \
    --threshold "${THRESHOLD}" \
    --block_topk_ratio "${BLOCK_TOPK_RATIO}" \
    \
    --score_stride 16 \
    --score_chunk_size 0 \
    --score_sample_kernel_size 7 \
    --score_norm 1.0 \
    \
    --teacher_rows 16 \
    --teacher_tail_rows 8 \
    --teacher_tokens_per_row 6 \
    \
    --teacher_head_chunk 2 \
    --teacher_key_chunk 1024 \
    \
    --positive_temperature 0.01 \
    \
    --teacher_kl_weight 0.45 \
    --teacher_l1_weight 0.10 \
    \
    --target_loss_weight 0.65 \
    \
    --aggregation_loss_weight 0.14 \
    --aggregation_cover_mass 0.32 \
    \
    --target_margin 0.12 \
    --target_joint_weight 0.90 \
    --target_query_tail_blocks 32 \
    \
    --negative_weight 0.025 \
    \
    --compression_loss_weight 0.0 \
    --compression_loss_weight_final 0.0 \
    \
    --budget_loss_weight 0.0 \
    --target_blocks_schedule "0:256" \
    \
    --bounded_delta_alpha "${BOUNDED_DELTA_ALPHA}" \
    \
    --ema_decay 0.999 \
    --max_grad_norm 0.04 \
    \
    --log_steps 10 \
    --save_steps "${SAVE_STEPS}"


# ==============================================================================
# 3. Verify final outputs
# ==============================================================================

echo
echo "=============================================================================="
echo "[3/3] Verify final checkpoints"
echo "=============================================================================="


if [[ ! -f "${OUT_PATH}" ]]; then

    echo "ERROR: final raw checkpoint not found:"
    echo "${OUT_PATH}"

    exit 1
fi


EMA_PATH="${OUT_PATH%.pt}_ema.pt"


if [[ ! -f "${EMA_PATH}" ]]; then

    echo "ERROR: final EMA checkpoint not found:"
    echo "${EMA_PATH}"

    exit 1
fi


python ft_scripts/conv_ruler/verify_conv_kernel.py \
    --path "${OUT_PATH}"


python ft_scripts/conv_ruler/verify_conv_kernel.py \
    --path "${EMA_PATH}"


# ==============================================================================
# Done
# ==============================================================================

echo
echo "=============================================================================="
echo "Hard continuation training complete"
echo "=============================================================================="

echo
echo "Anchor checkpoint:"
echo "${INIT_PATH}"

echo
echo "Dataset:"
echo "${SYNTH_DATA}"

echo
echo "Raw final:"
echo "${OUT_PATH}"

echo
echo "EMA final:"
echo "${EMA_PATH}"

echo
echo "Training state:"
echo "${OUT_PATH%.pt}_train_state.pt"

echo
echo "Training log:"
echo "${LOG_FILE}"


echo
echo "=============================================================================="
echo "Recommended checkpoints for RULER evaluation"
echo "=============================================================================="

for STEP in \
    500 \
    1000 \
    1500 \
    2000 \
    2500 \
    3000 \
    3500 \
    4000 \
    4500 \
    5000 \
    6000
do
    CKPT="${OUT_PATH%.pt}_step${STEP}.pt"

    if [[ -f "${CKPT}" ]]; then
        echo "${CKPT}"
    fi
done

echo "=============================================================================="