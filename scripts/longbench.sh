#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 MODEL TASK {xattn|conv|minference|flex|full} [extra args]" >&2
    exit 2
fi

MODEL="$1"
TASK="$2"
METHOD="$3"
shift 3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

exec python -u eval/LongBench/pred.py \
    --model "${MODEL}" \
    --task "${TASK}" \
    --method "${METHOD}" \
    "$@"
