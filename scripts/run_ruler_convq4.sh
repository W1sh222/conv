#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_ruler_convq4.sh [xattn|conv|minference|flex] [extra RULER args]
METHOD="${1:-conv}"
if [[ $# -gt 0 ]]; then shift; fi
case "${METHOD}" in
  xattn|conv|minference|flex) ;;
  *) echo "Unsupported method: ${METHOD} (expected xattn|conv|minference|flex)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}/eval/RULER/scripts"

exec bash ./runq64_1.sh qwen3-8b synthetic \
  --stride "${STRIDE:-8}" \
  --metric "${METHOD}" \
  --block_topk_ratio "${BLOCK_TOPK_RATIO:-0.65}" \
  "$@"
