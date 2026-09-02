#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_ruler_conv1.sh [xattn|conv|minference|flex] [extra run1.sh args]
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
case "${METHOD}" in
  xattn|conv|minference|flex) ;;
  *) echo "unsupported method: ${METHOD}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}/eval/RULER/scripts"

exec bash ./run1.sh llama3.1-8b-chat synthetic \
  --stride "${STRIDE:-8}" \
  --metric "${METHOD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO:-0.65}" \
  "$@"
