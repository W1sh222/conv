#!/usr/bin/env bash
# Small paired multi-task loss sweep; separate from existing experiment3--5 results.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_ROOT"
PYTHON_BIN="${PYTHON:-python}"
MODEL="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct"
SEQ_LENGTH=131072
NUM_SAMPLES=2
SEED=20261008
TASKS="qa_2,fwe,niah_single_1,niah_multikey_1"
WEIGHT=""
DATA=""
DATA_DIR=""
RAW_DATA_DIR=""
OUTPUT_ROOT="experiments5/kernel_task_results/llama128k_4tasks_2samples"
DRY_RUN=false
RESUME=false
FIXED_ONLY=false
EXTRA_ARGS=()
usage() {
    cat <<'EOF'
Usage: bash experiments5/run_kernel_task_sweep_llama128k.sh [options]
Defaults: Llama 128K, QA2/FWE/NIAH-S1/NIAH-MK1, 2 inputs/task,
six fixed V+diagonal kernels, Top-k=0.65, stride=8.
  --conv-weights FILE     Add a learned Llama 7x7 checkpoint (required unless --fixed-only)
  --fixed-only            Run only six untrained kernels (1,3,5,7,9,11)
  --resume                Reuse completed input-policy evaluations
  --num-samples N         Inputs per task, default 2
  --tasks CSV             Repository task keys, default qa_2,fwe,niah_single_1,niah_multikey_1
  --seq-length N          Context target, default 131072
  --data FILE             Existing combined converted observation.jsonl
  --data-dir DIR          Automatic prepared data directory
  --raw-data-dir DIR      Explicit existing Llama-templated RULER data (TASK/validation.jsonl)
  --output-root DIR       Preserve/reuse this path when resuming
  --model DIR             Matching Llama backbone
  --data-seed N           Generation seed, default 20261008
  --python BIN            Python from activated evaluation environment
  --dry-run               Print preparation and evaluation plans only
Other options forwarded to runner, e.g. --ratio 0.7, --stride 8.
Qwen step9750 has a different layer layout and cannot be applied to Llama.
EOF
}
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { printf 'Missing value: %s\n' "$1" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conv-weights) need_value "$@"; WEIGHT="$2"; shift 2 ;;
        --model) need_value "$@"; MODEL="$2"; shift 2 ;;
        --num-samples) need_value "$@"; NUM_SAMPLES="$2"; shift 2 ;;
        --tasks) need_value "$@"; TASKS="$2"; shift 2 ;;
        --seq-length) need_value "$@"; SEQ_LENGTH="$2"; shift 2 ;;
        --data-seed) need_value "$@"; SEED="$2"; shift 2 ;;
        --data) need_value "$@"; DATA="$2"; shift 2 ;;
        --data-dir) need_value "$@"; DATA_DIR="$2"; shift 2 ;;
        --raw-data-dir) need_value "$@"; RAW_DATA_DIR="$2"; shift 2 ;;
        --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        --fixed-only) FIXED_ONLY=true; shift ;;
        --resume) RESUME=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
if "$FIXED_ONLY" && [[ -n "$WEIGHT" ]]; then printf 'Choose --fixed-only OR --conv-weights.\n' >&2; exit 2; fi
if ! "$FIXED_ONLY" && [[ -z "$WEIGHT" ]]; then printf 'Supply a matching Llama --conv-weights, or --fixed-only. Qwen9750 is incompatible with Llama.\n' >&2; exit 2; fi
[[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ && "$SEQ_LENGTH" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]] || { printf 'Invalid numeric configuration\n' >&2; exit 2; }
if ! "$DRY_RUN" && ! "$FIXED_ONLY"; then
    "$PYTHON_BIN" experiments5/kernel_task_sweep/check_checkpoint.py --model "$MODEL" --conv-weights "$WEIGHT"
fi
if [[ -z "$DATA" ]]; then
    DATA_DIR="${DATA_DIR:-experiments5/kernel_task_data/llama_${SEQ_LENGTH}_${NUM_SAMPLES}samples_seed${SEED}}"
    DATA="$DATA_DIR/observation.jsonl"
    PREP=("$PYTHON_BIN" -u experiments5/kernel_task_sweep/prepare_data.py --model "$MODEL"
          --output "$DATA_DIR" --tasks "$TASKS" --num-samples "$NUM_SAMPLES" --seq-length "$SEQ_LENGTH" --seed "$SEED")
    if [[ -n "$RAW_DATA_DIR" ]]; then PREP+=(--raw-data-dir "$RAW_DATA_DIR"); fi
    if "$DRY_RUN"; then PREP+=(--dry-run); fi
    "${PREP[@]}"
fi
CMD=("$PYTHON_BIN" -u experiments5/kernel_task_sweep/run_sweep.py --model "$MODEL" --data "$DATA"
     --output "$OUTPUT_ROOT" --tasks "$TASKS" --samples-per-task "$NUM_SAMPLES" --seq-length "$SEQ_LENGTH"
     --kernel-sizes 1,3,5,7,9,11 --ratio 0.65 --stride 8 "${EXTRA_ARGS[@]}")
if "$FIXED_ONLY"; then CMD+=(--fixed-only); else CMD+=(--conv-weights "$WEIGHT"); fi
if "$RESUME"; then CMD+=(--resume); fi
if "$DRY_RUN"; then CMD+=(--dry-run); fi
"${CMD[@]}"
