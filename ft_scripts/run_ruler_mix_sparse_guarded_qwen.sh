#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B T0.65 balanced 64K-128K continuation framework.
#
# The previous 48K-64K step16000 EMA is treated as a quality anchor. Only its
# Conv weights are loaded; optimizer and scheduler state are intentionally
# reset. Every stage uses the same static YaRN factor-4 coordinates required by
# 128K inference, block_size=128, score_stride=8 and fixed Top-K ratio 0.65.
#
# Curriculum:
#   stage1: 48K-72K bridge       (protect the old RULER ranking)
#   stage2: 64K-96K extension
#   stage3: 96K-128K extension
#   stage4: 32K-128K consolidation (protect LongBench/64K transfer)
#   stage5: 8K-64K native-RoPE calibration (LongBench checkpoint)

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

BASE_INIT="${BASE_INIT:-${XATTN_ROOT}/qwen_weights/conv_qwen3/conv_kernel_7x7_qwen3_8b_ruler_mix_sparse_guarded_t065_multikey_qa2_48k64k_bf16_ema_step16000.pt}"
RUN_NAME="${RUN_NAME:-conv_qwen3_t065_64k128k_balanced_v3}"
WEIGHT_DIR="${XATTN_ROOT}/qwen_weights/${RUN_NAME}"
DATA_DIR="${NOLIMA_ROOT}/synth_train/${RUN_NAME}"
LOG_DIR="${WEIGHT_DIR}/logs"

MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
SHORT_LAYERS_PER_SAMPLE="${SHORT_LAYERS_PER_SAMPLE:-2}"
LONG_LAYERS_PER_SAMPLE="${LONG_LAYERS_PER_SAMPLE:-1}"
SAVE_STEPS="${SAVE_STEPS:-250}"
AUTO_RESUME="${AUTO_RESUME:-1}"

THRESHOLD=0.65
BLOCK_TOPK_RATIO=0.65
BLOCK_SIZE=128
SCORE_STRIDE=8
WEIGHT_MIN="${WEIGHT_MIN:--1.0}"
WEIGHT_MAX="${WEIGHT_MAX:-2.0}"

ROPE_SCALING_TYPE=yarn
ROPE_FACTOR=4.0
ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=32768
MAX_POSITION_EMBEDDINGS=131072

# RULER retention remains the main objective. QA1 and dense/general examples
# cover LongBench-like QA/summarization behavior without diluting multikey/QA2.
# The weights sum to exactly 1.0.
TASK_MIX="niah_single_1:0.005,niah_single_2:0.005,niah_single_3:0.005,niah_multikey_1:0.24,niah_multivalue:0.16,niah_multiquery:0.18,vt:0.06,cwe:0.08,fwe:0.05,qa_1:0.05,qa_2:0.12,dense_general:0.045"
LONGBENCH_TASK_MIX="niah_single_1:0.005,niah_single_2:0.005,niah_single_3:0.005,niah_multikey_1:0.18,niah_multivalue:0.12,niah_multiquery:0.14,vt:0.05,cwe:0.06,fwe:0.04,qa_1:0.11,qa_2:0.16,dense_general:0.125"

STAGE1_SAMPLES="${STAGE1_SAMPLES:-6000}"
STAGE2_SAMPLES="${STAGE2_SAMPLES:-8000}"
STAGE3_SAMPLES="${STAGE3_SAMPLES:-10000}"
STAGE4_SAMPLES="${STAGE4_SAMPLES:-8000}"
STAGE5_SAMPLES="${STAGE5_SAMPLES:-6000}"

STAGE1_STEPS="${STAGE1_STEPS:-6000}"
STAGE2_STEPS="${STAGE2_STEPS:-8000}"
STAGE3_STEPS="${STAGE3_STEPS:-10000}"
STAGE4_STEPS="${STAGE4_STEPS:-8000}"
STAGE5_STEPS="${STAGE5_STEPS:-6000}"

STAGE1_LR="${STAGE1_LR:-6e-6}"
STAGE2_LR="${STAGE2_LR:-4e-6}"
STAGE3_LR="${STAGE3_LR:-2.5e-6}"
STAGE4_LR="${STAGE4_LR:-1.5e-6}"
STAGE5_LR="${STAGE5_LR:-1e-6}"

STAGE1_DATA="${DATA_DIR}/stage1_48k72k_${STAGE1_SAMPLES}.jsonl"
STAGE2_DATA="${DATA_DIR}/stage2_64k96k_${STAGE2_SAMPLES}.jsonl"
STAGE3_DATA="${DATA_DIR}/stage3_96k128k_${STAGE3_SAMPLES}.jsonl"
STAGE4_DATA="${DATA_DIR}/stage4_32k128k_${STAGE4_SAMPLES}.jsonl"
STAGE5_DATA="${DATA_DIR}/stage5_longbench_native_8k64k_${STAGE5_SAMPLES}.jsonl"

STAGE1_OUT="${WEIGHT_DIR}/stage1_bridge_48k72k_t065_s8_yarn4_${MODEL_PRECISION}.pt"
STAGE2_OUT="${WEIGHT_DIR}/stage2_extend_64k96k_t065_s8_yarn4_${MODEL_PRECISION}.pt"
STAGE3_OUT="${WEIGHT_DIR}/stage3_extend_96k128k_t065_s8_yarn4_${MODEL_PRECISION}.pt"
FINAL_OUT="${WEIGHT_DIR}/conv_kernel_7x7_qwen3_t065_balanced_32k128k_s8_yarn4_${MODEL_PRECISION}.pt"
LONGBENCH_OUT="${WEIGHT_DIR}/conv_kernel_7x7_qwen3_t065_longbench_native_8k64k_s8_${MODEL_PRECISION}.pt"

mkdir -p "${WEIGHT_DIR}" "${DATA_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_qwen3_t065_64k128k_balanced_${MODEL_PRECISION}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Working directory: $(pwd)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "NOLIMA_ROOT=${NOLIMA_ROOT}"
echo "BASE_INIT=${BASE_INIT}"
echo "RUN_NAME=${RUN_NAME}"
echo "WEIGHT_DIR=${WEIGHT_DIR}"
echo "MODEL_PRECISION=${MODEL_PRECISION}"
echo "SHORT_LAYERS_PER_SAMPLE=${SHORT_LAYERS_PER_SAMPLE}"
echo "LONG_LAYERS_PER_SAMPLE=${LONG_LAYERS_PER_SAMPLE}"
echo "THRESHOLD=${THRESHOLD}"
echo "BLOCK_TOPK_RATIO=${BLOCK_TOPK_RATIO}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "SCORE_STRIDE=${SCORE_STRIDE}"
echo "ROPE_SCALING_TYPE=${ROPE_SCALING_TYPE}"
echo "ROPE_FACTOR=${ROPE_FACTOR}"
echo "MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS}"
echo "WEIGHT_RANGE=(${WEIGHT_MIN},${WEIGHT_MAX})"
echo "TASK_MIX=${TASK_MIX}"
echo "LONGBENCH_TASK_MIX=${LONGBENCH_TASK_MIX}"
echo "AUTO_RESUME=${AUTO_RESUME}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

test -d "${MODEL_PATH}" || { echo "model directory does not exist: ${MODEL_PATH}"; exit 1; }
test -d "${NOLIMA_ROOT}" || { echo "NoLiMa directory does not exist: ${NOLIMA_ROOT}"; exit 1; }
test -f "${BASE_INIT}" || { echo "quality-anchor checkpoint does not exist: ${BASE_INIT}"; exit 1; }

build_dataset() {
  local data_path="$1"
  local samples="$2"
  local min_length="$3"
  local max_length="$4"
  local distractors="$5"
  local position_mix="$6"
  local data_seed="$7"
  local task_mix="${8:-${TASK_MIX}}"
  local current_samples=0

  if [[ -f "${data_path}" ]]; then
    current_samples="$(wc -l < "${data_path}")"
  fi
  if [[ "${REBUILD_DATA:-0}" == "1" || "${current_samples}" -ne "${samples}" ]]; then
    local build_path="${data_path}.building"
    echo "[data] rebuilding ${data_path} current=${current_samples} expected=${samples}"
    python ft_scripts/sparse_ruler_qwen/build_ruler_mix_sft.py \
      --model "${MODEL_PATH}" \
      --nolima_root "${NOLIMA_ROOT}" \
      --out "${build_path}" \
      --haystack_subdirs rand_shuffle_long rand_shuffle \
      --num_samples "${samples}" \
      --min_seq_length "${min_length}" \
      --max_seq_length "${max_length}" \
      --num_distractor_needles "${distractors}" \
      --position_mix "${position_mix}" \
      --task_mix "${task_mix}" \
      --seed "${data_seed}"
    local built_samples
    built_samples="$(wc -l < "${build_path}")"
    test "${built_samples}" -eq "${samples}" || {
      echo "incomplete dataset: ${built_samples}/${samples}"
      exit 1
    }
    mv -f "${build_path}" "${data_path}"
  fi

  current_samples="$(wc -l < "${data_path}")"
  test "${current_samples}" -eq "${samples}" || {
    echo "dataset integrity failure: ${data_path} ${current_samples}/${samples}"
    exit 1
  }
}

run_stage() {
  local stage_name="$1"
  local data_path="$2"
  local init_path="$3"
  local out_path="$4"
  local min_length="$5"
  local max_length="$6"
  local train_steps="$7"
  local learning_rate="$8"
  local warmup_steps="$9"
  local layers_per_sample="${10}"
  local teacher_rows="${11}"
  local teacher_tail_rows="${12}"
  local target_tail_blocks="${13}"
  local teacher_kl_weight="${14}"
  local teacher_l1_weight="${15}"
  local topk_loss_weight="${16}"
  local target_loss_weight="${17}"
  local aggregation_loss_weight="${18}"
  local boundary_negatives="${19}"
  local train_seed="${20}"
  local min_lr_ratio="${21}"
  local ema_path="${out_path%.pt}_ema.pt"
  local state_path="${out_path%.pt}_train_state.pt"
  local resume_args=()

  if [[ -f "${ema_path}" ]]; then
    echo "[${stage_name}] completed checkpoint exists; skipping: ${ema_path}"
    return
  fi
  test -f "${init_path}" || {
    echo "[${stage_name}] initial checkpoint does not exist: ${init_path}"
    exit 1
  }
  if [[ "${AUTO_RESUME}" == "1" && -f "${state_path}" ]]; then
    resume_args+=(--resume_state "${state_path}")
    echo "[${stage_name}] auto-resuming ${state_path}"
  fi

  echo "================================================================================"
  echo "[${stage_name}] range=${min_length}-${max_length} steps=${train_steps} lr=${learning_rate} init=${init_path}"
  echo "[${stage_name}] data=${data_path} out=${out_path}"
  echo "================================================================================"

  python ft_scripts/sparse_ruler_qwen/train_conv_kernel_guarded_long.py \
    --model "${MODEL_PATH}" \
    --data "${data_path}" \
    --out "${out_path}" \
    --init_path "${init_path}" \
    "${resume_args[@]}" \
    --model_precision "${MODEL_PRECISION}" \
    --model_type qwen3 \
    --num_layers 36 \
    --num_heads 32 \
    --num_key_value_heads 8 \
    --kernel_size 7 \
    --layers_per_sample "${layers_per_sample}" \
    --min_seq_length "${min_length}" \
    --max_seq_length "${max_length}" \
    --steps "${train_steps}" \
    --lr "${learning_rate}" \
    --lr_schedule cosine \
    --warmup_steps "${warmup_steps}" \
    --min_lr_ratio "${min_lr_ratio}" \
    --seed "${train_seed}" \
    --rope_scaling_type "${ROPE_SCALING_TYPE}" \
    --rope_factor "${ROPE_FACTOR}" \
    --rope_original_max_position_embeddings "${ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS}" \
    --max_position_embeddings_override "${MAX_POSITION_EMBEDDINGS}" \
    --threshold "${THRESHOLD}" \
    --block_topk_ratio "${BLOCK_TOPK_RATIO}" \
    --block_size "${BLOCK_SIZE}" \
    --score_stride "${SCORE_STRIDE}" \
    --score_chunk_size 0 \
    --score_sample_kernel_size 7 \
    --score_norm 1.0 \
    --teacher_rows "${teacher_rows}" \
    --teacher_tail_rows "${teacher_tail_rows}" \
    --teacher_tokens_per_row 4 \
    --teacher_head_chunk 1 \
    --teacher_key_chunk 512 \
    --positive_temperature 0.02 \
    --teacher_kl_weight "${teacher_kl_weight}" \
    --teacher_l1_weight "${teacher_l1_weight}" \
    --topk_recall_loss_weight "${topk_loss_weight}" \
    --topk_train_ratio "${BLOCK_TOPK_RATIO}" \
    --topk_positive_mass 0.99 \
    --topk_boundary_negatives "${boundary_negatives}" \
    --topk_boundary_margin 0.03 \
    --target_loss_weight "${target_loss_weight}" \
    --aggregation_loss_weight "${aggregation_loss_weight}" \
    --aggregation_cover_mass 0.45 \
    --target_margin 0.12 \
    --target_joint_weight 0.90 \
    --target_query_tail_blocks "${target_tail_blocks}" \
    --negative_weight 0.03 \
    --compression_loss_weight 0.0 \
    --compression_loss_weight_final 0.0 \
    --budget_loss_weight 0.0 \
    --target_blocks_schedule "0:512" \
    --weight_min "${WEIGHT_MIN}" \
    --weight_max "${WEIGHT_MAX}" \
    --ema_decay 0.998 \
    --max_grad_norm 0.05 \
    --log_steps 10 \
    --save_steps "${SAVE_STEPS}"

  python ft_scripts/sparse_ruler_qwen/verify_qwen_conv_kernel.py --path "${out_path}"
  python ft_scripts/sparse_ruler_qwen/verify_qwen_conv_kernel.py --path "${ema_path}"
}

build_dataset "${STAGE1_DATA}" "${STAGE1_SAMPLES}" 49152 73728 96 \
  "uniform:0.58,edge:0.24,bimodal:0.18" 651101
run_stage stage1 "${STAGE1_DATA}" "${BASE_INIT}" "${STAGE1_OUT}" \
  49152 73728 "${STAGE1_STEPS}" "${STAGE1_LR}" 300 \
  "${SHORT_LAYERS_PER_SAMPLE}" 20 12 32 0.45 0.08 1.5 1.00 0.28 32 651011 0.20

build_dataset "${STAGE2_DATA}" "${STAGE2_SAMPLES}" 65536 98304 112 \
  "uniform:0.62,edge:0.22,bimodal:0.16" 651202
run_stage stage2 "${STAGE2_DATA}" "${STAGE1_OUT%.pt}_ema.pt" "${STAGE2_OUT}" \
  65536 98304 "${STAGE2_STEPS}" "${STAGE2_LR}" 250 \
  "${LONG_LAYERS_PER_SAMPLE}" 24 16 64 0.40 0.06 2.0 1.10 0.30 48 651022 0.15

build_dataset "${STAGE3_DATA}" "${STAGE3_SAMPLES}" 98304 131072 128 \
  "uniform:0.65,edge:0.22,bimodal:0.13" 651303
run_stage stage3 "${STAGE3_DATA}" "${STAGE2_OUT%.pt}_ema.pt" "${STAGE3_OUT}" \
  98304 131072 "${STAGE3_STEPS}" "${STAGE3_LR}" 250 \
  "${LONG_LAYERS_PER_SAMPLE}" 28 20 96 0.40 0.05 2.5 1.15 0.32 64 651033 0.10

# Final mixed-length pass: the lower LR and stronger teacher term are deliberate
# guards against repeating boundary-v2's RULER collapse.
build_dataset "${STAGE4_DATA}" "${STAGE4_SAMPLES}" 32768 131072 112 \
  "uniform:0.58,edge:0.25,bimodal:0.17" 651404
run_stage stage4 "${STAGE4_DATA}" "${STAGE3_OUT%.pt}_ema.pt" "${FINAL_OUT}" \
  32768 131072 "${STAGE4_STEPS}" "${STAGE4_LR}" 200 \
  "${LONG_LAYERS_PER_SAMPLE}" 24 16 64 0.45 0.06 2.0 1.00 0.30 48 651044 0.05

# Static YaRN is necessary for 128K but can reduce short-context model quality.
# Calibrate a separate LongBench checkpoint with native Qwen3 RoPE and a very
# small LR; the YaRN RULER checkpoint above remains unchanged.
ROPE_SCALING_TYPE=none
MAX_POSITION_EMBEDDINGS=0
build_dataset "${STAGE5_DATA}" "${STAGE5_SAMPLES}" 8192 65536 80 \
  "uniform:0.55,edge:0.25,bimodal:0.20" 651505 "${LONGBENCH_TASK_MIX}"
run_stage stage5 "${STAGE5_DATA}" "${FINAL_OUT%.pt}_ema.pt" "${LONGBENCH_OUT}" \
  8192 65536 "${STAGE5_STEPS}" "${STAGE5_LR}" 150 \
  "${SHORT_LAYERS_PER_SAMPLE}" 20 12 32 0.50 0.08 1.5 0.85 0.25 32 651055 0.10

echo "================================================================================"
echo "Qwen3 T0.65 64K-128K balanced continuation complete."
echo "Experiment directory: ${WEIGHT_DIR}"
echo "Final raw checkpoint: ${FINAL_OUT}"
echo "Final EMA checkpoint: ${FINAL_OUT%.pt}_ema.pt"
echo "Final training state: ${FINAL_OUT%.pt}_train_state.pt"
echo "LongBench native raw checkpoint: ${LONGBENCH_OUT}"
echo "LongBench native EMA checkpoint: ${LONGBENCH_OUT%.pt}_ema.pt"
echo "Recommended RULER 64K candidate: ${STAGE1_OUT%.pt}_ema.pt"
echo "Recommended RULER 96K candidate: ${STAGE2_OUT%.pt}_ema.pt"
echo "Recommended RULER 128K candidate: ${STAGE3_OUT%.pt}_ema.pt"
echo "Recommended 64K-128K mixed candidate: ${FINAL_OUT%.pt}_ema.pt"
echo "Recommended LongBench candidate: ${LONGBENCH_OUT%.pt}_ema.pt"
