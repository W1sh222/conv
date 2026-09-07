#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_ruler_convq1.sh [xattn|conv|minference|flex|full] [extra runq1.sh args]
# Run only qa_1 with: RULER_TASKS=qa_1 bash scripts/run_ruler_convq1.sh conv
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
case "${METHOD}" in
  xattn|conv|minference|flex|full) ;;
  *) echo "unsupported method: ${METHOD}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}/eval/RULER/scripts"

exec bash ./runq1.sh qwen3-8b synthetic \
  --stride "${STRIDE:-8}" \
  --metric "${METHOD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO:-0.65}" \
  "$@"
