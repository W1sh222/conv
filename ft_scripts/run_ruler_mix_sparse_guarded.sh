#!/usr/bin/env bash
set -euo pipefail

# Llama-3.1-8B sparse-Conv T0.65 scratch curriculum.
#
# This experiment never loads an older learned Conv checkpoint. Stage 1 starts
# from the exact identity/no-convolution kernel, then stages 2-4 continue only
# from checkpoints produced by this run. The four stages progressively cover
# 24K-32K, 32K-48K, 48K-64K, and finally the complete 24K-64K range.
#
# Inference-matched invariants:
#   block_size=128, score_stride=8, topk_ratio=0.65, kernel_size=7.

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

RUN_NAME="${RUN_NAME:-conv_llama_t065_scratch_curriculum_v1}"
WEIGHT_DIR="${XATTN_ROOT}/llama_weights/${RUN_NAME}"
DATA_DIR="${NOLIMA_ROOT}/synth_train/${RUN_NAME}"
LOG_DIR="${WEIGHT_DIR}/logs"

MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-2}"
SAVE_STEPS="${SAVE_STEPS:-250}"
AUTO_RESUME="${AUTO_RESUME:-1}"

THRESHOLD=0.65
BLOCK_TOPK_RATIO=0.65
BLOCK_SIZE=128
SCORE_STRIDE=8
WEIGHT_MIN="${WEIGHT_MIN:--1.0}"
WEIGHT_MAX="${WEIGHT_MAX:-2.0}"

# Historical runs were strongest on multikey and QA2-heavy mixtures. Keep that
# emphasis, add QA1 coverage, and retain 5% dense/general data for LongBench.
# The weights sum to exactly 1.0.
TASK_MIX="niah_single_1:0.005,niah_single_2:0.005,niah_single_3:0.005,niah_multikey_1:0.25,niah_multivalue:0.15,niah_multiquery:0.17,vt:0.06,cwe:0.08,fwe:0.05,qa_1:0.06,qa_2:0.115,dense_general:0.05"

STAGE1_SAMPLES="${STAGE1_SAMPLES:-8000}"
STAGE2_SAMPLES="${STAGE2_SAMPLES:-9000}"
STAGE3_SAMPLES="${STAGE3_SAMPLES:-12000}"
STAGE4_SAMPLES="${STAGE4_SAMPLES:-6000}"

STAGE1_STEPS="${STAGE1_STEPS:-8000}"
STAGE2_STEPS="${STAGE2_STEPS:-9000}"
STAGE3_STEPS="${STAGE3_STEPS:-12000}"
STAGE4_STEPS="${STAGE4_STEPS:-6000}"

STAGE1_LR="${STAGE1_LR:-3e-5}"
STAGE2_LR="${STAGE2_LR:-1.8e-5}"
STAGE3_LR="${STAGE3_LR:-1e-5}"
STAGE4_LR="${STAGE4_LR:-5e-6}"

STAGE1_DATA="${DATA_DIR}/stage1_24k32k_${STAGE1_SAMPLES}.jsonl"
STAGE2_DATA="${DATA_DIR}/stage2_32k48k_${STAGE2_SAMPLES}.jsonl"
STAGE3_DATA="${DATA_DIR}/stage3_48k64k_${STAGE3_SAMPLES}.jsonl"
STAGE4_DATA="${DATA_DIR}/stage4_24k64k_${STAGE4_SAMPLES}.jsonl"

STAGE1_OUT="${WEIGHT_DIR}/stage1_24k32k_t065_${MODEL_PRECISION}.pt"
STAGE2_OUT="${WEIGHT_DIR}/stage2_32k48k_t065_${MODEL_PRECISION}.pt"
STAGE3_OUT="${WEIGHT_DIR}/stage3_48k64k_t065_${MODEL_PRECISION}.pt"
FINAL_OUT="${WEIGHT_DIR}/conv_kernel_7x7_llama_t065_scratch_curriculum_24k64k_${MODEL_PRECISION}.pt"

mkdir -p "${WEIGHT_DIR}" "${DATA_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_t065_scratch_curriculum_${MODEL_PRECISION}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Working directory: $(pwd)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "NOLIMA_ROOT=${NOLIMA_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "WEIGHT_DIR=${WEIGHT_DIR}"
echo "MODEL_PRECISION=${MODEL_PRECISION}"
echo "LAYERS_PER_SAMPLE=${LAYERS_PER_SAMPLE}"
echo "THRESHOLD=${THRESHOLD}"
echo "BLOCK_TOPK_RATIO=${BLOCK_TOPK_RATIO}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "SCORE_STRIDE=${SCORE_STRIDE}"
echo "WEIGHT_RANGE=(${WEIGHT_MIN},${WEIGHT_MAX})"
echo "TASK_MIX=${TASK_MIX}"
echo "AUTO_RESUME=${AUTO_RESUME}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

test -d "${MODEL_PATH}" || { echo "model directory does not exist: ${MODEL_PATH}"; exit 1; }
test -d "${NOLIMA_ROOT}" || { echo "NoLiMa directory does not exist: ${NOLIMA_ROOT}"; exit 1; }

build_dataset() {
  local data_path="$1"
  local samples="$2"
  local min_length="$3"
  local max_length="$4"
  local distractors="$5"
  local position_mix="$6"
  local data_seed="$7"
  local current_samples=0

  if [[ -f "${data_path}" ]]; then
    current_samples="$(wc -l < "${data_path}")"
  fi
  if [[ "${REBUILD_DATA:-0}" == "1" || "${current_samples}" -ne "${samples}" ]]; then
    local build_path="${data_path}.building"
    echo "[data] rebuilding ${data_path} current=${current_samples} expected=${samples}"
    python ft_scripts/sparse_ruler/build_ruler_mix_sft.py \
      --model "${MODEL_PATH}" \
      --nolima_root "${NOLIMA_ROOT}" \
      --out "${build_path}" \
      --haystack_subdirs rand_shuffle_long rand_shuffle \
      --num_samples "${samples}" \
      --min_seq_length "${min_length}" \
      --max_seq_length "${max_length}" \
      --num_distractor_needles "${distractors}" \
      --position_mix "${position_mix}" \
      --task_mix "${TASK_MIX}" \
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
  local teacher_rows="${10}"
  local teacher_tail_rows="${11}"
  local target_tail_blocks="${12}"
  local teacher_kl_weight="${13}"
  local teacher_l1_weight="${14}"
  local topk_loss_weight="${15}"
  local target_loss_weight="${16}"
  local aggregation_loss_weight="${17}"
  local boundary_negatives="${18}"
  local train_seed="${19}"
  local min_lr_ratio="${20}"
  local ema_path="${out_path%.pt}_ema.pt"
  local state_path="${out_path%.pt}_train_state.pt"
  local resume_args=()

  if [[ -f "${ema_path}" ]]; then
    echo "[${stage_name}] completed checkpoint exists; skipping: ${ema_path}"
    return
  fi
  if [[ "${init_path}" != "identity" ]]; then
    test -f "${init_path}" || {
      echo "[${stage_name}] initial checkpoint does not exist: ${init_path}"
      exit 1
    }
  fi
  if [[ "${AUTO_RESUME}" == "1" && -f "${state_path}" ]]; then
    resume_args+=(--resume_state "${state_path}")
    echo "[${stage_name}] auto-resuming ${state_path}"
  fi

  echo "================================================================================"
  echo "[${stage_name}] range=${min_length}-${max_length} steps=${train_steps} lr=${learning_rate} init=${init_path}"
  echo "[${stage_name}] data=${data_path} out=${out_path}"
  echo "================================================================================"

  python ft_scripts/sparse_ruler/train_conv_kernel_guarded_long.py \
    --model "${MODEL_PATH}" \
    --data "${data_path}" \
    --out "${out_path}" \
    --init_path "${init_path}" \
    "${resume_args[@]}" \
    --model_precision "${MODEL_PRECISION}" \
    --model_type llama \
    --num_layers 32 \
    --num_heads 32 \
    --num_key_value_heads 8 \
    --kernel_size 7 \
    --layers_per_sample "${LAYERS_PER_SAMPLE}" \
    --min_seq_length "${min_length}" \
    --max_seq_length "${max_length}" \
    --steps "${train_steps}" \
    --lr "${learning_rate}" \
    --lr_schedule cosine \
    --warmup_steps "${warmup_steps}" \
    --min_lr_ratio "${min_lr_ratio}" \
    --seed "${train_seed}" \
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
    --target_blocks_schedule "0:256" \
    --weight_min "${WEIGHT_MIN}" \
    --weight_max "${WEIGHT_MAX}" \
    --ema_decay 0.998 \
    --max_grad_norm 0.05 \
    --log_steps 10 \
    --save_steps "${SAVE_STEPS}"

  python ft_scripts/sparse_ruler/verify_conv_kernel.py \
    --path "${out_path}" --num_layers 32 --num_heads 32 --kernel_size 7
  python ft_scripts/sparse_ruler/verify_conv_kernel.py \
    --path "${ema_path}" --num_layers 32 --num_heads 32 --kernel_size 7
}

# Build each shard immediately before its stage. Dataset files are published
# atomically and safely reused on restart.
build_dataset "${STAGE1_DATA}" "${STAGE1_SAMPLES}" 24576 32768 64 \
  "uniform:0.55,edge:0.25,bimodal:0.20" 650101
# Stage 1 is the only scratch initialization: no learned .pt is read here.
run_stage stage1 "${STAGE1_DATA}" identity "${STAGE1_OUT}" \
  24576 32768 "${STAGE1_STEPS}" "${STAGE1_LR}" 400 \
  20 12 32 0.45 0.08 2.0 0.75 0.25 32 650011 0.20

build_dataset "${STAGE2_DATA}" "${STAGE2_SAMPLES}" 32768 49152 80 \
  "uniform:0.60,edge:0.20,bimodal:0.20" 650202
run_stage stage2 "${STAGE2_DATA}" "${STAGE1_OUT%.pt}_ema.pt" "${STAGE2_OUT}" \
  32768 49152 "${STAGE2_STEPS}" "${STAGE2_LR}" 350 \
  20 14 48 0.40 0.06 2.75 0.95 0.28 40 650022 0.15

build_dataset "${STAGE3_DATA}" "${STAGE3_SAMPLES}" 49152 65536 96 \
  "uniform:0.65,edge:0.20,bimodal:0.15" 650303
run_stage stage3 "${STAGE3_DATA}" "${STAGE2_OUT%.pt}_ema.pt" "${STAGE3_OUT}" \
  49152 65536 "${STAGE3_STEPS}" "${STAGE3_LR}" 300 \
  24 16 64 0.35 0.05 3.5 1.15 0.30 48 650033 0.10

# Low-LR mixed-length consolidation reduces catastrophic specialization to
# only 48K-64K while retaining the hard T0.65 boundary learned in stage 3.
build_dataset "${STAGE4_DATA}" "${STAGE4_SAMPLES}" 24576 65536 96 \
  "uniform:0.60,edge:0.22,bimodal:0.18" 650404
run_stage stage4 "${STAGE4_DATA}" "${STAGE3_OUT%.pt}_ema.pt" "${FINAL_OUT}" \
  24576 65536 "${STAGE4_STEPS}" "${STAGE4_LR}" 200 \
  24 16 64 0.40 0.06 3.0 1.00 0.28 48 650044 0.05

echo "================================================================================"
echo "Llama T0.65 scratch curriculum complete."
echo "Experiment directory: ${WEIGHT_DIR}"
echo "Final raw checkpoint: ${FINAL_OUT}"
echo "Final EMA checkpoint: ${FINAL_OUT%.pt}_ema.pt"
echo "Final training state: ${FINAL_OUT%.pt}_train_state.pt"
echo "Recommended first evaluation candidate: ${FINAL_OUT%.pt}_ema.pt"
echo "Also compare the stage-3 EMA: ${STAGE3_OUT%.pt}_ema.pt"
