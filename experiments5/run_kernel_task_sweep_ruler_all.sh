#!/usr/bin/env bash
# Extend the fixed-kernel NLL analysis without modifying the four-task results.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."
MODE=remaining
NUM_SAMPLES=2
SEQ_LENGTH=131072
SEED=20261008
OUTPUT=""
DATA_DIR=""
LEARNED=false
EXTRA=()
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { printf 'Missing value: %s\n' "$1" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-tasks) MODE=all; shift ;;
        --remaining-tasks) MODE=remaining; shift ;;
        --num-samples) need_value "$@"; NUM_SAMPLES="$2"; shift 2 ;;
        --seq-length) need_value "$@"; SEQ_LENGTH="$2"; shift 2 ;;
        --data-seed) need_value "$@"; SEED="$2"; shift 2 ;;
        --output-root) need_value "$@"; OUTPUT="$2"; shift 2 ;;
        --data-dir) need_value "$@"; DATA_DIR="$2"; shift 2 ;;
        --conv-weights) need_value "$@"; LEARNED=true; EXTRA+=("$1" "$2"); shift 2 ;;
        --tasks) printf 'Use --all-tasks/--remaining-tasks, or the original launcher for a custom task list.\n' >&2; exit 2 ;;
        --help|-h)
            printf '%s\n' 'Usage: bash experiments5/run_kernel_task_sweep_ruler_all.sh [options]' \
              'Default: remaining 9 RULER tasks, Llama 128K, two inputs/task, fixed kernels 1/3/5/7/9/11.' \
              '--all-tasks: run all 13 tasks in a separate directory (does not import the old four).' \
              '--remaining-tasks: run only the nine not in the existing four-task sweep.' \
              '--resume: preserve completed policies and skip recorded numerical failures.' \
              'Other options pass to run_kernel_task_sweep_llama128k.sh; see its --help.'
            exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done
[[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ && "$SEQ_LENGTH" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]] || { printf 'Invalid numeric configuration\n' >&2; exit 2; }
if [[ "$MODE" == all ]]; then
    TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2"
    GROUP=all13tasks
else
    TASKS="niah_single_2,niah_single_3,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,qa_1"
    GROUP=remaining9tasks
fi
if [[ "$SEQ_LENGTH" == 131072 ]]; then LENGTH=128k; else LENGTH="$SEQ_LENGTH"; fi
OUTPUT="${OUTPUT:-experiments5/kernel_task_results/llama${LENGTH}_${GROUP}_${NUM_SAMPLES}samples}"
DATA_DIR="${DATA_DIR:-experiments5/kernel_task_data/llama_${SEQ_LENGTH}_${GROUP}_${NUM_SAMPLES}samples_seed${SEED}}"
CMD=(bash "$SCRIPT_DIR/run_kernel_task_sweep_llama128k.sh"
     --tasks "$TASKS" --num-samples "$NUM_SAMPLES" --seq-length "$SEQ_LENGTH" --data-seed "$SEED"
     --data-dir "$DATA_DIR" --output-root "$OUTPUT")
if ! "$LEARNED"; then CMD+=(--fixed-only); fi
CMD+=("${EXTRA[@]}")
printf 'Tasks: %s\nData directory: %s\nResults: %s\n' "$TASKS" "$DATA_DIR" "$OUTPUT"
"${CMD[@]}"
