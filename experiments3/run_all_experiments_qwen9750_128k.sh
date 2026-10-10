#!/usr/bin/env bash
# Qwen step9750 128K diagnostics, with opt-in interrupted-run recovery.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_ROOT"

MODEL="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B"
WEIGHT="$REPO_ROOT/xattn/qwen_weights/conv_qwen3_t065_64k128k_balanced_v3/stage3_extend_96k128k_t065_s8_yarn4_bf16_ema_step9750.pt"
PYTHON_BIN="${PYTHON:-python}"
SEQ_LENGTH=131072
NUM_SAMPLES=5
DATA_SEED=20261008
DATA=""
DATA_DIR=""
SAMPLE_INDICES=""
OUTPUT_ROOT="experiments3/experiments3_5/qwen9750_vt128k_5samples"
DRY_RUN=false
RESUME=false
EXTRA_ARGS=(--rope-factor 4 --rope-original-length 32768)

usage() {
    cat <<'EOF'
Usage: bash experiments3/run_all_experiments_qwen9750_128k.sh [options]
Defaults: Qwen3-8B, step9750 EMA checkpoint, five new 128K VT inputs,
seed 20261008, Top-k=0.65, stride=8, layer16/head8/query=last.
  --model DIR             Llama model directory
  --conv-weights FILE     Matching Qwen convolution checkpoint
  --data FILE             Reuse an existing Qwen-templated observation.jsonl
  --data-dir DIR          Directory for automatic Qwen VT data preparation
  --seq-length N          Generated context length (default: 131072)
  --num-samples N         Generated sample count (default: 5)
  --data-seed N           Data generation seed
  --sample-indices CSV    Inputs to evaluate (default: all generated inputs)
  --output-root DIR       Results directory (reuse exactly the same path with --resume)
  --python BIN            Python executable (default: active environment)
  --dry-run               Print preparation/run configurations only
  --resume                Reuse completed cases/trials after an interruption
  --help                  Show this help
Defaults include YaRN factor=4, original context=32768; keep these on resume.
Other options are forwarded, e.g. --ratio 0.7, --no-plots, --max-candidates 20.
This is fixed-budget Top-k NLL analysis; it does not run Top-p=0.92 evaluation.
EOF
}
need_value() {
    [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) need_value "$@"; MODEL="$2"; shift 2 ;;
        --conv-weights) need_value "$@"; WEIGHT="$2"; shift 2 ;;
        --data) need_value "$@"; DATA="$2"; shift 2 ;;
        --data-dir) need_value "$@"; DATA_DIR="$2"; shift 2 ;;
        --seq-length) need_value "$@"; SEQ_LENGTH="$2"; shift 2 ;;
        --num-samples) need_value "$@"; NUM_SAMPLES="$2"; shift 2 ;;
        --data-seed) need_value "$@"; DATA_SEED="$2"; shift 2 ;;
        --sample-indices) need_value "$@"; SAMPLE_INDICES="$2"; shift 2 ;;
        --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; EXTRA_ARGS+=(--resume); shift ;;
        --help|-h) usage; exit 0 ;;
        --learned-policy-name|--learned-policy-name=*|--output|--output=*)
            printf 'Output/policy naming is managed by this launcher; use --output-root.\n' >&2; exit 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
[[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ && "$SEQ_LENGTH" =~ ^[1-9][0-9]*$ && "$DATA_SEED" =~ ^[0-9]+$ ]] || {
    printf 'num-samples/seq-length must be positive integers; data-seed must be nonnegative.\n' >&2; exit 2;
}
if [[ -z "$SAMPLE_INDICES" ]]; then
    for ((i=0;i<NUM_SAMPLES;i++)); do SAMPLE_INDICES+="${SAMPLE_INDICES:+,}$i"; done
fi
command -v "$PYTHON_BIN" >/dev/null || { printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2; exit 2; }
if ! "$DRY_RUN"; then
    [[ -f "$WEIGHT" ]] || { printf 'Qwen checkpoint not found: %s\n' "$WEIGHT" >&2; exit 2; }
    if [[ -e "$OUTPUT_ROOT" && "$RESUME" == false ]]; then
        [[ -d "$OUTPUT_ROOT" && -z "$(ls -A -- "$OUTPUT_ROOT")" ]] || {
            printf 'Use a new/empty output root: %s\n' "$OUTPUT_ROOT" >&2; exit 2;
        }
    fi
fi

if [[ -z "$DATA" ]]; then
    DATA_DIR="${DATA_DIR:-experiments3/datasets/qwen9750_vt${SEQ_LENGTH}_${NUM_SAMPLES}samples_seed${DATA_SEED}}"
    DATA="$DATA_DIR/observation.jsonl"
    if [[ -f "$DATA" && "$DRY_RUN" == false ]]; then
        # Only reuse automatic data if its provenance matches this model/config.
        "$PYTHON_BIN" - "$DATA_DIR/pipeline.json" "$MODEL" "$SEQ_LENGTH" "$NUM_SAMPLES" "$DATA_SEED" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit('Missing data provenance; use a fresh --data-dir or explicitly supply --data')
d=json.loads(p.read_text());a=d['arguments']
if (Path(a['model']).resolve()!=Path(sys.argv[2]).resolve() or
    [a['seq_length'],a['num_samples'],a['seed']]!=list(map(int,sys.argv[3:])) or
    d.get('model_template')!='qwen3' or d.get('status') not in ('data_ready','complete')):
    raise SystemExit('Existing automatic data has different model/config/status; use a fresh --data-dir')
PY
    else
        PREP=("$PYTHON_BIN" -u experiments/run_ruler_observation.py --model "$MODEL"
              --model-template qwen3 --rope-factor 4 --rope-original-length 32768 --seq-length "$SEQ_LENGTH" --num-samples "$NUM_SAMPLES"
              --seed "$DATA_SEED" --prepare-only --output "$DATA_DIR")
        printf '[Qwen VT preparation] '; printf '%q ' "${PREP[@]}"; printf '\n'
        if ! "$DRY_RUN"; then "${PREP[@]}"; fi
    fi
fi
CMD=(bash "$SCRIPT_DIR/run_all_experiments.sh" --python "$PYTHON_BIN" --data "$DATA"
     --model "$MODEL" --conv-weights "$WEIGHT" --sample-indices "$SAMPLE_INDICES"
     --learned-policy-name learned_9750 --output-root "$OUTPUT_ROOT" "${EXTRA_ARGS[@]}")
if "$DRY_RUN"; then CMD+=(--dry-run); fi
"${CMD[@]}"
