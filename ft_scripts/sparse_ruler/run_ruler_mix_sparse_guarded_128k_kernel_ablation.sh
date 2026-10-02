#!/usr/bin/env bash
set -euo pipefail

# Shared 96K-128K T0.7 training recipe for the K=5/K=9 ablations.
# Wrapper scripts set KERNEL_SIZE and EXPERIMENT_NAME.  Each experiment starts
# from its own vertical+diagonal KxK kernel and trains from step 0 to 4000;
# no incompatible 7x7 checkpoint is loaded.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

KERNEL_SIZE="${KERNEL_SIZE:?KERNEL_SIZE must be 5 or 9}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
if [[ "${KERNEL_SIZE}" != "5" && "${KERNEL_SIZE}" != "9" ]]; then
  echo "KERNEL_SIZE must be 5 or 9, got ${KERNEL_SIZE}" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct}"
NOLIMA_ROOT="${NOLIMA_ROOT:-/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/dllm/data/NoLiMa}"
XATTN_ROOT="${XATTN_ROOT:-/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn}"
WEIGHT_DIR="${WEIGHT_DIR:-${XATTN_ROOT}/conv_weights/kernel${KERNEL_SIZE}_scratch_t07_96k128k}"

# The trainer expands this sentinel to a vertical+diagonal kernel of the
# requested size.  INIT_PATH can still be overridden with a matching-shape
# checkpoint for a later continuation experiment.
INIT_PATH="${INIT_PATH:-scratch_vertical_diag}"
MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
DATA_SAMPLES="${DATA_SAMPLES:-10000}"
TRAIN_STEPS="${TRAIN_STEPS:-4000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-1}"
LR="${LR:-5e-7}"
WARMUP_STEPS="${WARMUP_STEPS:-400}"
THRESHOLD=0.7
BLOCK_TOPK_RATIO=0.7

SYNTH_DATA="${SYNTH_DATA:-${NOLIMA_ROOT}/synth_train/ruler_mix_sparse_t07_96k128k_kernel_ablation_${DATA_SAMPLES}.jsonl}"
OUT_PATH="${OUT_PATH:-${WEIGHT_DIR}/conv_kernel_${KERNEL_SIZE}x${KERNEL_SIZE}_ruler_mix_sparse_guarded_long_t07_96k128k_from_scratch_step4000_${MODEL_PRECISION}.pt}"
LOG_DIR="${LOG_DIR:-./ft_scripts/sparse_ruler/logs/kernel${KERNEL_SIZE}_ablation}"
mkdir -p "${LOG_DIR}" "${WEIGHT_DIR}" "$(dirname "${SYNTH_DATA}")" "$(dirname "${OUT_PATH}")"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}_${MODEL_PRECISION}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "KERNEL_SIZE=${KERNEL_SIZE}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "INIT_PATH=${INIT_PATH}"
echo "SYNTH_DATA=${SYNTH_DATA}"
echo "OUT_PATH=${OUT_PATH}"
echo "TRAIN_STEPS=${TRAIN_STEPS} LR=${LR} WARMUP_STEPS=${WARMUP_STEPS}"
echo "THRESHOLD=${THRESHOLD} BLOCK_TOPK_RATIO=${BLOCK_TOPK_RATIO}"

test -d "${MODEL_PATH}" || { echo "model directory does not exist: ${MODEL_PATH}"; exit 1; }
test -d "${NOLIMA_ROOT}" || { echo "NoLiMa directory does not exist: ${NOLIMA_ROOT}"; exit 1; }
# Both ablations use the same deterministic data recipe and can share this
# JSONL.  A complete line-count check prevents a partial build from training.
CURRENT_SAMPLES=0
if [[ -f "${SYNTH_DATA}" ]]; then CURRENT_SAMPLES="$(wc -l < "${SYNTH_DATA}")"; fi
if [[ "${REBUILD_DATA:-0}" == "1" || "${CURRENT_SAMPLES}" -ne "${DATA_SAMPLES}" ]]; then
  BUILD_PATH="${SYNTH_DATA}.building"
  python ft_scripts/sparse_ruler/build_ruler_mix_sft.py \
    --model "${MODEL_PATH}" \
    --nolima_root "${NOLIMA_ROOT}" \
    --out "${BUILD_PATH}" \
    --haystack_subdirs rand_shuffle_long rand_shuffle \
    --num_samples "${DATA_SAMPLES}" \
    --min_seq_length 98304 \
    --max_seq_length 131072 \
    --num_distractor_needles 64 \
    --position_mix "uniform:0.45,edge:0.30,bimodal:0.25" \
    --task_mix "niah_single_1:0.015,niah_single_2:0.020,niah_single_3:0.015,niah_multikey_1:0.150,niah_multikey_2:0.220,niah_multikey_3:0.060,niah_multivalue:0.080,niah_multiquery:0.080,vt:0.040,cwe:0.030,fwe:0.050,qa_1:0.140,qa_2:0.080,dense_general:0.020" \
    --seed 1800400
  BUILT_SAMPLES="$(wc -l < "${BUILD_PATH}")"
  test "${BUILT_SAMPLES}" -eq "${DATA_SAMPLES}" || { echo "incomplete dataset: ${BUILT_SAMPLES}/${DATA_SAMPLES}"; exit 1; }
  mv -f "${BUILD_PATH}" "${SYNTH_DATA}"
fi
test "$(wc -l < "${SYNTH_DATA}")" -eq "${DATA_SAMPLES}" || { echo "dataset integrity failure"; exit 1; }

python ft_scripts/sparse_ruler/train_conv_kernel_guarded_long.py \
  --model "${MODEL_PATH}" \
  --data "${SYNTH_DATA}" \
  --out "${OUT_PATH}" \
  --init_path "${INIT_PATH}" \
  --model_type llama \
  --model_precision "${MODEL_PRECISION}" \
  --num_layers 32 \
  --num_heads 32 \
  --num_key_value_heads 8 \
  --kernel_size "${KERNEL_SIZE}" \
  --layers_per_sample "${LAYERS_PER_SAMPLE}" \
  --min_seq_length 98304 \
  --max_seq_length 131072 \
  --steps "${TRAIN_STEPS}" \
  --lr "${LR}" \
  --lr_schedule cosine \
  --warmup_steps "${WARMUP_STEPS}" \
  --min_lr_ratio 0.25 \
  --seed 1800400 \
  --rope_scaling_type none \
  --rope_factor 4.0 \
  --rope_original_max_position_embeddings 32768 \
  --max_position_embeddings_override 131072 \
  --threshold "${THRESHOLD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO}" \
  --block_size 128 \
  --score_stride 8 \
  --score_chunk_size 0 \
  --score_sample_kernel_size "${KERNEL_SIZE}" \
  --score_norm 1.0 \
  --teacher_rows 8 \
  --teacher_tail_rows 8 \
  --teacher_tokens_per_row 4 \
  --teacher_head_chunk 1 \
  --teacher_key_chunk 512 \
  --positive_temperature 0.015 \
  --teacher_kl_weight 0.50 \
  --teacher_l1_weight 0.08 \
  --topk_recall_loss_weight 0.75 \
  --topk_train_ratio 0.70 \
  --topk_positive_mass 0.99 \
  --topk_boundary_negatives 48 \
  --topk_boundary_margin 0.025 \
  --target_loss_weight 0.55 \
  --aggregation_loss_weight 0.15 \
  --aggregation_cover_mass 0.40 \
  --target_margin 0.08 \
  --target_joint_weight 0.85 \
  --target_query_tail_blocks 64 \
  --negative_weight 0.015 \
  --compression_loss_weight 0.0 \
  --compression_loss_weight_final 0.0 \
  --budget_loss_weight 0.0 \
  --target_blocks_schedule "0:768" \
  --bounded_delta_alpha 0.04 \
  --ema_decay 0.9997 \
  --max_grad_norm 0.03 \
  --log_steps 10 \
  --save_steps "${SAVE_STEPS}"

python ft_scripts/sparse_ruler/verify_conv_kernel.py --path "${OUT_PATH}" --kernel_size "${KERNEL_SIZE}"
python ft_scripts/sparse_ruler/verify_conv_kernel.py --path "${OUT_PATH%.pt}_ema.pt" --kernel_size "${KERNEL_SIZE}"

echo "Kernel-${KERNEL_SIZE} ablation training complete."
echo "Raw checkpoint: ${OUT_PATH}"
echo "EMA checkpoint: ${OUT_PATH%.pt}_ema.pt"
