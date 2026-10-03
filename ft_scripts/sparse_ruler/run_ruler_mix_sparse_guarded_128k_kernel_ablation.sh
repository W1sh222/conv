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

# Command-line overrides are intentionally accepted after the .sh path.  The
# environment-variable form remains supported for compatibility, but users
# do not need export statements anymore.
REBUILD_DATA="${REBUILD_DATA:-0}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild-data)
      REBUILD_DATA=1
      shift
      ;;
    --rebuild-cache)
      REBUILD_CACHE=1
      shift
      ;;
    --data-samples)
      [[ $# -ge 2 ]] || { echo "--data-samples requires a value" >&2; exit 2; }
      DATA_SAMPLES="$2"
      shift 2
      ;;
    --steps)
      [[ $# -ge 2 ]] || { echo "--steps requires a value" >&2; exit 2; }
      TRAIN_STEPS="$2"
      shift 2
      ;;
    --lr)
      [[ $# -ge 2 ]] || { echo "--lr requires a value" >&2; exit 2; }
      LR="$2"
      shift 2
      ;;
    --save-steps)
      [[ $# -ge 2 ]] || { echo "--save-steps requires a value" >&2; exit 2; }
      SAVE_STEPS="$2"
      shift 2
      ;;
    --layers-per-sample)
      [[ $# -ge 2 ]] || { echo "--layers-per-sample requires a value" >&2; exit 2; }
      LAYERS_PER_SAMPLE="$2"
      shift 2
      ;;
    --out)
      [[ $# -ge 2 ]] || { echo "--out requires a path" >&2; exit 2; }
      OUT_PATH="$2"
      shift 2
      ;;
    --cache-dir)
      [[ $# -ge 2 ]] || { echo "--cache-dir requires a path" >&2; exit 2; }
      DATASET_CACHE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash run_ruler_mix_sparse_guarded_128k_kernel{5,9}_ablation.sh [options]
  --rebuild-data       rebuild the shared JSONL dataset
  --rebuild-cache      remove this run's exact Arrow cache directory
  --data-samples N     dataset size (default: 10000)
  --steps N             training steps (default: 4000)
  --lr X               learning rate
  --save-steps N        checkpoint interval
  --layers-per-sample N
  --out PATH            output checkpoint path
  --cache-dir PATH      dataset cache directory
USAGE
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

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
# Keep Arrow caches isolated per kernel.  The previous failed run left a
# truncated cache; reusing it causes pyarrow's "message body" read error even
# after the JSONL has been rebuilt.  Set REBUILD_CACHE=1 to clear this exact
# run-specific directory when recovering from another interrupted import.
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${WEIGHT_DIR}/hf_datasets_cache_k${KERNEL_SIZE}_v2}"
if [[ "${REBUILD_CACHE:-0}" == "1" ]]; then
  case "${DATASET_CACHE_DIR}" in
    "${WEIGHT_DIR}"/*) rm -rf -- "${DATASET_CACHE_DIR}" ;;
    *) echo "refusing to remove cache outside WEIGHT_DIR: ${DATASET_CACHE_DIR}" >&2; exit 2 ;;
  esac
fi
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DATASET_CACHE_DIR}}"
LOG_DIR="${LOG_DIR:-./ft_scripts/sparse_ruler/logs/kernel${KERNEL_SIZE}_ablation}"
mkdir -p "${LOG_DIR}" "${WEIGHT_DIR}" "$(dirname "${SYNTH_DATA}")" "$(dirname "${OUT_PATH}")" "${DATASET_CACHE_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}_${MODEL_PRECISION}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "KERNEL_SIZE=${KERNEL_SIZE}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "INIT_PATH=${INIT_PATH}"
echo "SYNTH_DATA=${SYNTH_DATA}"
echo "DATASET_CACHE_DIR=${DATASET_CACHE_DIR}"
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

# Validate every JSON record before datasets/pyarrow sees it.  This catches a
# truncated final line or malformed record independently of the Arrow cache.
python ft_scripts/sparse_ruler/check_jsonl_integrity.py \
  --path "${SYNTH_DATA}" \
  --expected "${DATA_SAMPLES}"

python ft_scripts/sparse_ruler/train_conv_kernel_guarded_long.py \
  --model "${MODEL_PATH}" \
  --data "${SYNTH_DATA}" \
  --cache_dir "${DATASET_CACHE_DIR}" \
  --lazy_jsonl \
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
