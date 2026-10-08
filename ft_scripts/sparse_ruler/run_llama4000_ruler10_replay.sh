#!/usr/bin/env bash
set -euo pipefail
# New branch from the evaluated continuation EMA, not the earlier *_step4000.pt.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CONV_TOPP_SELECTOR=positive
BASE=/inspire/hdd/global_user/gexinmu-253108100065
MODEL_PATH="${MODEL_PATH:-${BASE}/Resources/models/LLMs/Llama-3.1-8B-Instruct}"
NOLIMA_ROOT="${NOLIMA_ROOT:-${BASE}/Repos/fuyicheng_workshop/dllm/data/NoLiMa}"
INIT_PATH="${INIT_PATH:-${REPO_ROOT}/xattn/conv_weights/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_96k128k_continue_from4000_v1_bf16_ema.pt}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/xattn/conv_weights/llama4000_ruler10_replay_topp092_v1}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
LR="${LR:-2e-7}"
TOPP="${TOPP:-0.92}"
MAIN_SAMPLES="${MAIN_SAMPLES:-4000}"
REPLAY_SAMPLES="${REPLAY_SAMPLES:-2000}"
REPLAY_DATA="${REPLAY_DATA:-}"
RESUME_STATE="${RESUME_STATE:-}"
DRY_RUN=0
usage() {
  cat <<'EOF'
Usage: bash ft_scripts/sparse_ruler/run_llama4000_ruler10_replay.sh [options]
  --init PATH          Evaluated 4000-continuation EMA
  --run-dir PATH       New output directory
  --model PATH         Llama model directory
  --steps N            Total steps (default 2000)
  --lr VALUE           Default 2e-7
  --topp VALUE         Default 0.92; use the same value in evaluation
  --replay-data PATH   Independent training JSONL with messages or text
  --resume-state PATH  This new run's own training state
  --dry-run            Print plan and command without loading model/building data
Default replay uses fresh synthetic QA/aggregation/general tasks, NOT LongBench test data.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --init|--run-dir|--model|--steps|--lr|--topp|--replay-data|--resume-state)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      case "$1" in
        --init) INIT_PATH="$2" ;; --run-dir) RUN_DIR="$2" ;; --model) MODEL_PATH="$2" ;;
        --steps) TRAIN_STEPS="$2" ;; --lr) LR="$2" ;; --topp) TOPP="$2" ;;
        --replay-data) REPLAY_DATA="$2" ;; --resume-state) RESUME_STATE="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "${TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "--steps must be a positive integer" >&2; exit 2; }
WARMUP_STEPS=$(( TRAIN_STEPS < 2000 ? TRAIN_STEPS / 10 : 200 ))
MAIN_DATA="${RUN_DIR}/data/ruler10_112k128k_seed128092.jsonl"
DEFAULT_REPLAY="${RUN_DIR}/data/replay_8k64k_seed640092.jsonl"
REPLAY_DATA="${REPLAY_DATA:-${DEFAULT_REPLAY}}"
OUT_PATH="${RUN_DIR}/conv_kernel_7x7_llama4000_ruler10_replay.pt"
MAIN_MIX="niah_single_1:0.03,niah_single_2:0.03,niah_single_3:0.03,niah_multikey_1:0.15,niah_multivalue:0.10,niah_multiquery:0.10,vt:0.15,cwe:0.20,fwe:0.10,qa_2:0.11"
REPLAY_MIX="niah_single_1:0.02,niah_single_2:0.02,niah_single_3:0.02,niah_multikey_1:0.05,niah_multikey_2:0.02,niah_multikey_3:0.02,niah_multivalue:0.10,niah_multiquery:0.10,vt:0.08,cwe:0.08,fwe:0.08,qa_1:0.10,qa_2:0.11,dense_general:0.20"
CMD=(python ft_scripts/sparse_ruler/train_conv_kernel_guarded_long.py
  --model "${MODEL_PATH}" --model_type llama --model_precision bf16
  --num_layers 32 --num_heads 32 --num_key_value_heads 8 --kernel_size 7
  --data "${MAIN_DATA}" --lazy_jsonl --replay_data "${REPLAY_DATA}" --replay_every 4
  --replay_min_seq_length 8192 --replay_max_seq_length 65536
  --init_path "${INIT_PATH}" --out "${OUT_PATH}"
  --min_seq_length 114688 --max_seq_length 131072 --layers_per_sample 1
  --steps "${TRAIN_STEPS}" --lr "${LR}" --lr_schedule cosine --warmup_steps "${WARMUP_STEPS}" --min_lr_ratio 0.25
  --seed 128092 --rope_scaling_type none --max_position_embeddings_override 131072
  --threshold "${TOPP}" --block_size 128 --score_stride 8 --score_chunk_size 0
  --score_sample_kernel_size 7 --score_norm 1.0
  --teacher_rows 16 --teacher_tail_rows 8 --teacher_tokens_per_row 4
  --teacher_head_chunk 1 --teacher_key_chunk 512 --positive_temperature 0.015
  --teacher_kl_weight 0.6 --teacher_l1_weight 0.10 --positive_teacher_weight 0.20
  --topk_recall_loss_weight 0.25 --topk_train_ratio 0.70 --topk_positive_mass 0.99
  --topk_boundary_negatives 32 --topk_boundary_margin 0.02
  --target_loss_weight 0.45 --aggregation_loss_weight 0.30 --aggregation_cover_mass 0.40
  --target_margin 0.06 --target_joint_weight 0.80 --target_query_tail_blocks 32
  --negative_weight 0.02 --compression_loss_weight 0 --compression_loss_weight_final 0
  --budget_loss_weight 0 --target_blocks_schedule "0:1024"
  --replay_anchor_weight 1.0 --kernel_anchor_weight 0.10 --bounded_delta_alpha 0.02
  --ema_decay 0.999 --max_grad_norm 0.02 --log_steps 10 --save_steps 250)
if [[ -n "${RESUME_STATE}" ]]; then CMD+=(--resume_state "${RESUME_STATE}"); fi
echo "Main tasks: ${MAIN_MIX}"
echo "Replay: 25% of updates; ${REPLAY_DATA}"
echo "Anchor: ${INIT_PATH}; selector=positive_mass_v1; Top-p=${TOPP}"
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN}" == 1 ]]; then exit 0; fi
test -d "${MODEL_PATH}" && test -f "${INIT_PATH}" || { echo "Missing model or input EMA" >&2; exit 1; }
if [[ -z "${RESUME_STATE}" && -f "${OUT_PATH%.pt}_train_state.pt" ]]; then
  echo "Run already exists; use --resume-state for this run, or choose a new --run-dir" >&2; exit 1
fi
if [[ -n "${RESUME_STATE}" ]]; then test -f "${RESUME_STATE}"; fi
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs"
exec > >(tee -a "${RUN_DIR}/logs/train.log") 2>&1
build_data() {
  local path="$1" samples="$2" min_len="$3" max_len="$4" mix="$5" seed="$6"
  if [[ -f "${path}" ]]; then
    [[ "$(wc -l < "${path}")" -eq "${samples}" ]] || { echo "Incomplete dataset: ${path}" >&2; exit 1; }
    return
  fi
  python ft_scripts/sparse_ruler/build_ruler_mix_sft.py \
    --model "${MODEL_PATH}" --nolima_root "${NOLIMA_ROOT}" --out "${path}.building" \
    --num_samples "${samples}" --min_seq_length "${min_len}" --max_seq_length "${max_len}" \
    --num_distractor_needles 64 --cwe_num_words 10 --position_mix 'uniform:0.55,edge:0.25,bimodal:0.20' \
    --task_mix "${mix}" --seed "${seed}"
  [[ "$(wc -l < "${path}.building")" -eq "${samples}" ]]
  mv "${path}.building" "${path}"
}
build_data "${MAIN_DATA}" "${MAIN_SAMPLES}" 114688 131072 "${MAIN_MIX}" 128092
if [[ "${REPLAY_DATA}" == "${DEFAULT_REPLAY}" ]]; then
  build_data "${REPLAY_DATA}" "${REPLAY_SAMPLES}" 8192 65536 "${REPLAY_MIX}" 640092
else
  test -s "${REPLAY_DATA}" || { echo "Missing replay training data" >&2; exit 1; }
fi
"${CMD[@]}"
python ft_scripts/sparse_ruler/verify_conv_kernel.py --path "${OUT_PATH%.pt}_ema.pt"
echo "Evaluate EMA checkpoints against the input anchor before accepting a replacement."
