#!/usr/bin/env bash
set -euo pipefail

# Fresh origin-kernel Qwen3 training: LongBench first, RULER replay second.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B}"
NOLIMA_ROOT="${NOLIMA_ROOT:-/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/dllm/data/NoLiMa}"
XATTN_ROOT="${XATTN_ROOT:-/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn}"
RUN_NAME="${RUN_NAME:-conv_qwen3_origin_longbench_v1}"
MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
DATA_SAMPLES="${DATA_SAMPLES:-10000}"
RULER_SAMPLES="${RULER_SAMPLES:-4000}"
PHASE1_STEPS="${PHASE1_STEPS:-4500}"
PHASE2_STEPS="${PHASE2_STEPS:-1800}"
PHASE1_LR="${PHASE1_LR:-2e-5}"
PHASE2_LR="${PHASE2_LR:-3e-6}"
PHASE1_WARMUP="${PHASE1_WARMUP:-350}"
PHASE2_WARMUP="${PHASE2_WARMUP:-180}"
LAYERS_PER_SAMPLE="${LAYERS_PER_SAMPLE:-1}"
SAVE_STEPS="${SAVE_STEPS:-250}"
AUTO_RESUME="${AUTO_RESUME:-1}"
REBUILD_DATA="${REBUILD_DATA:-0}"

LONG_TASK_MIX="niah_single_1:0.005,niah_single_2:0.005,niah_single_3:0.005,niah_multikey_1:0.06,niah_multikey_2:0.03,niah_multikey_3:0.015,niah_multivalue:0.08,niah_multiquery:0.06,vt:0.07,cwe:0.07,fwe:0.07,qa_1:0.20,qa_2:0.18,dense_general:0.165"
RULER_TASK_MIX="niah_single_1:0.01,niah_single_2:0.01,niah_single_3:0.01,niah_multikey_1:0.22,niah_multikey_2:0.15,niah_multikey_3:0.10,niah_multivalue:0.15,niah_multiquery:0.15,vt:0.05,cwe:0.04,fwe:0.03,qa_1:0.03,qa_2:0.03,dense_general:0.02"

usage() {
  cat <<'USAGE'
Usage: bash ft_scripts/run_ruler_qwen_origin_longbench.sh [options]
  --run_name NAME --samples N --ruler_samples N
  --phase1_steps N --phase2_steps N --phase1_lr LR --phase2_lr LR
  --rebuild_data --no_auto_resume
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --samples) DATA_SAMPLES="$2"; shift 2 ;;
    --ruler_samples) RULER_SAMPLES="$2"; shift 2 ;;
    --phase1_steps) PHASE1_STEPS="$2"; shift 2 ;;
    --phase2_steps) PHASE2_STEPS="$2"; shift 2 ;;
    --phase1_lr) PHASE1_LR="$2"; shift 2 ;;
    --phase2_lr) PHASE2_LR="$2"; shift 2 ;;
    --rebuild_data) REBUILD_DATA=1; shift ;;
    --no_auto_resume) AUTO_RESUME=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

WEIGHT_DIR="${XATTN_ROOT}/qwen_weights/${RUN_NAME}"
DATA_DIR="${NOLIMA_ROOT}/synth_train/${RUN_NAME}"
LOG_DIR="${WEIGHT_DIR}/logs"
ORIGIN_INIT="${WEIGHT_DIR}/origin_vertical_diag_36x32_7x7.pt"
P1_DATA="${DATA_DIR}/longbench_native_8k64k_${DATA_SAMPLES}.jsonl"
P2_DATA="${DATA_DIR}/ruler_yarn_96k128k_${RULER_SAMPLES}.jsonl"
P1_OUT="${WEIGHT_DIR}/conv_kernel_7x7_qwen3_origin_longbench_native_8k64k_s8_t065_${MODEL_PRECISION}.pt"
P2_OUT="${WEIGHT_DIR}/conv_kernel_7x7_qwen3_origin_longbench_ruler96k128k_s8_t065_${MODEL_PRECISION}.pt"
P1_STATE="${P1_OUT%.pt}_train_state.pt"
P2_STATE="${P2_OUT%.pt}_train_state.pt"
P1_EMA="${P1_OUT%.pt}_ema.pt"
P2_EMA="${P2_OUT%.pt}_ema.pt"
mkdir -p "${WEIGHT_DIR}" "${DATA_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/train_${RUN_NAME}_${MODEL_PRECISION}.log") 2>&1
echo "[origin-longbench] model=${MODEL_PATH} run=${RUN_NAME} topk=0.65 stride=8 block=128"
echo "[origin-longbench] p1=${P1_OUT} steps=${PHASE1_STEPS} lr=${PHASE1_LR}"
echo "[origin-longbench] p2=${P2_OUT} steps=${PHASE2_STEPS} lr=${PHASE2_LR}"
test -d "${MODEL_PATH}" || { echo "model directory does not exist: ${MODEL_PATH}"; exit 1; }
test -d "${NOLIMA_ROOT}" || { echo "NoLiMa directory does not exist: ${NOLIMA_ROOT}"; exit 1; }

if [[ ! -f "${ORIGIN_INIT}" ]]; then
  python ft_scripts/sparse_ruler_qwen_origin_longbench/make_origin_checkpoint.py \
    --out "${ORIGIN_INIT}" --layers 36 --heads 32 --kernel-size 7
fi

build_data() {
  local out="$1" samples="$2" lo="$3" hi="$4" needles="$5" mix="$6" seed="$7" current=0
  [[ -f "${out}" ]] && current="$(wc -l < "${out}")"
  if [[ "${REBUILD_DATA}" == 1 || "${current}" -ne "${samples}" ]]; then
    local tmp="${out}.building"
    python ft_scripts/sparse_ruler_qwen/build_ruler_mix_sft.py --model "${MODEL_PATH}" \
      --nolima_root "${NOLIMA_ROOT}" --out "${tmp}" --haystack_subdirs rand_shuffle_long rand_shuffle \
      --num_samples "${samples}" --min_seq_length "${lo}" --max_seq_length "${hi}" \
      --num_distractor_needles "${needles}" --position_mix "uniform:0.60,edge:0.20,bimodal:0.20" \
      --task_mix "${mix}" --seed "${seed}"
    [[ "$(wc -l < "${tmp}")" -eq "${samples}" ]] || { echo "incomplete dataset ${tmp}"; exit 1; }
    mv -f "${tmp}" "${out}"
  fi
  [[ "$(wc -l < "${out}")" -eq "${samples}" ]] || { echo "dataset integrity failure ${out}"; exit 1; }
}

build_data "${P1_DATA}" "${DATA_SAMPLES}" 8192 65536 48 "${LONG_TASK_MIX}" 651701
build_data "${P2_DATA}" "${RULER_SAMPLES}" 98304 131072 80 "${RULER_TASK_MIX}" 651702

run_phase() {
  local tag="$1" data="$2" init="$3" out="$4" state="$5" ema="$6"
  local lo="$7" hi="$8" steps="$9" lr="${10}" warm="${11}" rope="${12}" rope_max="${13}"
  local resume=()
  [[ -f "${ema}" ]] && { echo "[${tag}] EMA exists, skip: ${ema}"; return; }
  if [[ "${AUTO_RESUME}" == 1 && -f "${state}" ]]; then
    resume+=(--resume_state "${state}")
    echo "[${tag}] resuming ${state}"
  fi
  python ft_scripts/sparse_ruler_qwen/train_conv_kernel_guarded_long.py \
    --model "${MODEL_PATH}" --data "${data}" --out "${out}" --init_path "${init}" "${resume[@]}" \
    --model_precision "${MODEL_PRECISION}" --model_type qwen3 --num_layers 36 --num_heads 32 \
    --num_key_value_heads 8 --kernel_size 7 --layers_per_sample "${LAYERS_PER_SAMPLE}" \
    --min_seq_length "${lo}" --max_seq_length "${hi}" --steps "${steps}" --lr "${lr}" \
    --lr_schedule cosine --warmup_steps "${warm}" --min_lr_ratio 0.20 --seed 651703 \
    --rope_scaling_type "${rope}" --rope_factor 4.0 --rope_original_max_position_embeddings 32768 \
    --max_position_embeddings_override "${rope_max}" --threshold 0.65 --block_topk_ratio 0.65 \
    --block_size 128 --score_stride 8 --score_chunk_size 0 --score_sample_kernel_size 7 --score_norm 1.0 \
    --teacher_rows 16 --teacher_tail_rows 8 --teacher_tokens_per_row 4 --teacher_head_chunk 2 \
    --teacher_key_chunk 1024 --positive_temperature 0.01 --teacher_kl_weight 0.70 --teacher_l1_weight 0.20 \
    --target_loss_weight 0.25 --aggregation_loss_weight 0.08 --aggregation_cover_mass 0.30 \
    --target_margin 0.08 --target_joint_weight 0.75 --target_query_tail_blocks 20 --negative_weight 0.01 \
    --compression_loss_weight 0.0 --compression_loss_weight_final 0.0 --budget_loss_weight 0.0 \
    --target_blocks_schedule "0:256" --bounded_delta_alpha 0.10 --ema_decay 0.997 \
    --max_grad_norm 0.08 --log_steps 10 --save_steps "${SAVE_STEPS}"
}

run_phase phase1_longbench_native "${P1_DATA}" "${ORIGIN_INIT}" "${P1_OUT}" \
  "${P1_STATE}" "${P1_EMA}" 8192 65536 "${PHASE1_STEPS}" "${PHASE1_LR}" \
  "${PHASE1_WARMUP}" none 0
run_phase phase2_ruler_replay "${P2_DATA}" "${P1_EMA}" "${P2_OUT}" \
  "${P2_STATE}" "${P2_EMA}" 98304 131072 "${PHASE2_STEPS}" "${PHASE2_LR}" \
  "${PHASE2_WARMUP}" yarn 131072
echo "[origin-longbench] LongBench candidate: ${P1_EMA}"
echo "[origin-longbench] RULER-protected candidate: ${P2_EMA}"
