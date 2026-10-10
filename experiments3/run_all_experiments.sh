#!/usr/bin/env bash
# Run experiments 3, 4 and 5 sequentially in the active evaluation environment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_ROOT"

DATA="output/ruler_observation/vt_32k_seed42/observation.jsonl"
OUTPUT_ROOT="output/experiments3_5/qwen5000_$(date +%Y%m%d_%H%M%S)_$$"
MODEL="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B"
WEIGHT="xattn/qwen_weights/conv_qwen3_t065_longbench_stage4_v2/conv_kernel_7x7_qwen3_t065_longbench_replay_native_8k64k_s8_bf16_ema_step5000.pt"
PYTHON_BIN="${PYTHON:-python}"
DRY_RUN=false
RESUME=false
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash experiments3/run_all_experiments.sh [options]
  --data FILE         Existing observation.jsonl (default: vt_32k_seed42)
  --output-root DIR   New/empty directory containing experiment3/4/5 and logs
  --model DIR         Qwen3 or Llama model path (default: Qwen3)
  --conv-weights FILE Matching convolution weights (default: Qwen step5000)
  --python BIN        Python executable from the active evaluation environment
  --dry-run          Print all three configurations; do not load models/write results
  --resume           Continue the same output root; reuse completed cases/trials
  --help             Show this help
Other options are forwarded to all three runners, e.g. --sample-indices 0,1,2,
--ratio 0.65, --stride 8, --no-plots, --rope-factor 4.
Use one diagnostic layer/head/query location because experiment5 requires this.
Activate fyc_qwen before running. Default ratio=0.65, stride=8, sample index=0.
EOF
}

need_value() {
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) need_value "$@"; DATA="$2"; shift 2 ;;
        --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
        --model) need_value "$@"; MODEL="$2"; shift 2 ;;
        --conv-weights) need_value "$@"; WEIGHT="$2"; shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; EXTRA_ARGS+=(--resume); shift ;;
        --help|-h) usage; exit 0 ;;
        --output|--output=*) printf 'Use --output-root instead of --output.\n' >&2; exit 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

command -v "$PYTHON_BIN" >/dev/null || { printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2; exit 2; }
if ! "$DRY_RUN"; then
    [[ -f "$DATA" ]] || { printf 'Data file not found: %s\n' "$DATA" >&2; exit 2; }
    [[ -f "$WEIGHT" ]] || { printf 'Weight file not found: %s\n' "$WEIGHT" >&2; exit 2; }
    if [[ -e "$OUTPUT_ROOT" && "$RESUME" == false ]]; then
        [[ -d "$OUTPUT_ROOT" && -z "$(ls -A -- "$OUTPUT_ROOT")" ]] || {
            printf 'Output root must be a new/empty directory: %s\n' "$OUTPUT_ROOT" >&2; exit 2;
        }
    fi
    mkdir -p -- "$OUTPUT_ROOT"
fi

RUNNERS=(experiments3/run_ranking_utility.py experiments4/run_added_removed.py experiments5/run_neighborhood_ablation.py)
for i in 0 1 2; do
    NUMBER=$((i + 3))
    CMD=("$PYTHON_BIN" -u "${RUNNERS[$i]}" --data "$DATA" --model "$MODEL"
         --conv-weights "$WEIGHT" --ratio 0.65 --stride 8
         --output "$OUTPUT_ROOT/experiment$NUMBER" "${EXTRA_ARGS[@]}")
    printf '\n[Experiment %s] ' "$NUMBER"
    printf '%q ' "${CMD[@]}"
    printf '\n'
    if "$DRY_RUN"; then
        "${CMD[@]}" --dry-run
    else
        "${CMD[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/experiment$NUMBER.log"
    fi
done
if "$DRY_RUN"; then
    printf '\nAll three configuration checks completed; no inference performed.\n'
else
    printf '\nAll three experiments completed. Results: %s\n' "$OUTPUT_ROOT"
fi
