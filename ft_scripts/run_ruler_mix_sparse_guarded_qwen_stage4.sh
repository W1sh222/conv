#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B LongBench-focused Stage 4 (v2).
#
# This is intentionally a separate continuation entry point.  Stages 1-3 of
# conv_qwen3_t065_64k128k_balanced_v3 are left untouched; the input is the
# last requested Stage-3 EMA (step 9250).  This branch is deliberately
# LongBench-focused: it is a separate native-RoPE 8K-64K checkpoint and must
# not replace the YaRN 96K-128K RULER checkpoint.
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
RUN_NAME="${RUN_NAME:-conv_qwen3_t065_longbench_stage4_v2}"
WEIGHT_DIR="${XATTN_ROOT}/qwen_weights/${RUN_NAME}"
DATA_DIR="${NOLIMA_ROOT}/synth_train/${RUN_NAME}"
LOG_DIR="${WEIGHT_DIR}/logs"

MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
DATA_SAMPLES="${DATA_SAMPLES:-12000}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
LR="${LR:-1.5e-7}"
WARMUP_STEPS="${WARMUP_STEPS:-400}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-1}"
SAVE_STEPS="${SAVE_STEPS:-250}"
AUTO_RESUME="${AUTO_RESUME:-1}"

DATA_PATH="${DATA_DIR}/stage4_longbench_replay_native_8k64k_${DATA_SAMPLES}.jsonl"
OUT_PATH="${WEIGHT_DIR}/conv_kernel_7x7_qwen3_t065_longbench_replay_native_8k64k_s8_${MODEL_PRECISION}.pt"
STATE_PATH="${OUT_PATH%.pt}_train_state.pt"
EMA_PATH="${OUT_PATH%.pt}_ema.pt"
LOG_FILE="${LOG_DIR}/train_qwen3_t065_longbench_stage4_v2_${MODEL_PRECISION}.log"

# Synthetic RULER-Mix is still the only local training-data interface, so use
# its QA/dense tasks as the LongBench surrogate and retain 25% multikey replay.
# The replay is important: otherwise a native short-context pass destroys the
# ranking learned by the 9250 anchor.  The weights sum to 1.0.
TASK_MIX="niah_single_1:0.005,niah_single_2:0.005,niah_single_3:0.005,niah_multikey_1:0.10,niah_multikey_2:0.10,niah_multikey_3:0.05,niah_multivalue:0.10,niah_multiquery:0.10,vt:0.04,cwe:0.04,fwe:0.03,qa_1:0.18,qa_2:0.15,dense_general:0.095"

# Command-line arguments intentionally come after the script name, so a run
# can be reproduced without relying on shell environment assignments.  The
# old environment-variable interface remains supported for compatibility.
usage() {
  cat <<'USAGE'
Usage: bash run_ruler_mix_sparse_guarded_qwen_stage4.sh [options]

Options:
  --stage3_init PATH       Stage-3 EMA checkpoint used as the fixed anchor
  --run_name NAME          Output/checkpoint directory name
  --model_precision P      bf16 (default) or fp32
  --samples N              Number of synthetic training samples
  --steps N                Optimizer steps
  --lr VALUE               Initial learning rate
  --warmup_steps N         Cosine-schedule warmup steps
  --layers_per_sample N    Number of sampled transformer layers
  --save_steps N           Checkpoint interval
  --task_mix MIX           Override synthetic task mixture
  --model_path PATH        Qwen3 model directory
  --nolima_root PATH       NoLiMa data root
  --data PATH              Explicit JSONL dataset path
  --out PATH               Explicit output checkpoint path
  --no_auto_resume         Do not resume from an existing training-state file
  --rebuild_data           Rebuild the synthetic dataset before training
  -h, --help               Show this help
USAGE
}

EXPLICIT_DATA_PATH=""
EXPLICIT_OUT_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage3_init|--init_path)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      STAGE3_INIT="$2"; shift 2 ;;
    --run_name)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      RUN_NAME="$2"; shift 2 ;;
    --model_precision)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      MODEL_PRECISION="$2"; shift 2 ;;
    --samples|--data_samples)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      DATA_SAMPLES="$2"; shift 2 ;;
    --steps)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      TRAIN_STEPS="$2"; shift 2 ;;
    --lr)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      LR="$2"; shift 2 ;;
    --warmup_steps)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      WARMUP_STEPS="$2"; shift 2 ;;
    --layers_per_sample)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      LAYERS_PER_SAMPLE="$2"; shift 2 ;;
    --save_steps)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      SAVE_STEPS="$2"; shift 2 ;;
    --task_mix)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      TASK_MIX="$2"; shift 2 ;;
    --model_path)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      MODEL_PATH="$2"; shift 2 ;;
    --nolima_root)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      NOLIMA_ROOT="$2"; shift 2 ;;
    --data)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      EXPLICIT_DATA_PATH="$2"; shift 2 ;;
    --out)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      EXPLICIT_OUT_PATH="$2"; shift 2 ;;
    --no_auto_resume)
      AUTO_RESUME=0; shift ;;
    --rebuild_data)
      REBUILD_DATA=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

# Recompute paths after CLI overrides (especially --run_name and --samples).
WEIGHT_DIR="${XATTN_ROOT}/qwen_weights/${RUN_NAME}"
DATA_DIR="${NOLIMA_ROOT}/synth_train/${RUN_NAME}"
LOG_DIR="${WEIGHT_DIR}/logs"
DATA_PATH="${EXPLICIT_DATA_PATH:-${DATA_DIR}/stage4_longbench_replay_native_8k64k_${DATA_SAMPLES}.jsonl}"
OUT_PATH="${EXPLICIT_OUT_PATH:-${WEIGHT_DIR}/conv_kernel_7x7_qwen3_t065_longbench_replay_native_8k64k_s8_${MODEL_PRECISION}.pt}"
STATE_PATH="${OUT_PATH%.pt}_train_state.pt"
EMA_PATH="${OUT_PATH%.pt}_ema.pt"
LOG_FILE="${LOG_DIR}/train_qwen3_t065_longbench_stage4_v2_${MODEL_PRECISION}.log"

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
case "$(basename "${STAGE3_INIT}")" in
  *_step9250.pt) ;;
  *)
    echo "Stage4 v2 must start from the Stage-3 EMA at step 9250; got: ${STAGE3_INIT}" >&2
    exit 1
    ;;
esac

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
  --min_lr_ratio 0.25 \
  --seed 651704 \
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
  --teacher_kl_weight 0.70 \
  --teacher_l1_weight 0.10 \
  --topk_recall_loss_weight 0.30 \
  --topk_train_ratio 0.65 \
  --topk_positive_mass 0.95 \
  --topk_boundary_negatives 16 \
  --topk_boundary_margin 0.015 \
  --target_loss_weight 0.35 \
  --aggregation_loss_weight 0.10 \
  --aggregation_cover_mass 0.32 \
  --target_margin 0.08 \
  --target_joint_weight 0.60 \
  --target_query_tail_blocks 24 \
  --negative_weight 0.01 \
  --compression_loss_weight 0.0 \
  --compression_loss_weight_final 0.0 \
  --budget_loss_weight 0.0 \
  --target_blocks_schedule "0:512" \
  --bounded_delta_alpha 0.02 \
  --weight_min -1.0 \
  --weight_max 2.0 \
  --ema_decay 0.9995 \
  --max_grad_norm 0.02 \
  --log_steps 10 \
  --save_steps "${SAVE_STEPS}"

python ft_scripts/sparse_ruler_qwen/verify_qwen_conv_kernel.py --path "${OUT_PATH}"
python ft_scripts/sparse_ruler_qwen/verify_qwen_conv_kernel.py --path "${EMA_PATH}"

echo "Qwen corrective Stage 4 complete."
echo "Raw checkpoint: ${OUT_PATH}"
echo "EMA checkpoint: ${EMA_PATH}"
echo "Training state: ${STATE_PATH}"
