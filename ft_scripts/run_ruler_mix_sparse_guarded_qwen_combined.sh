#!/usr/bin/env bash
set -euo pipefail

# One-command Qwen continuation:
#   1) preserve/run the original Stage 1-3 curriculum to Stage-3 step 9250;
#   2) continue from that exact Stage-3 EMA with the corrective native 8K-64K
#      Stage 4 in run_ruler_mix_sparse_guarded_qwen_stage4.sh.
#
# Existing checkpoints and train_state files are reused automatically.  The
# legacy mixed Stage 4/5 in run_ruler_mix_sparse_guarded_qwen.sh are disabled
# through STOP_AFTER_STAGE3 so this combined entry point has one unambiguous
# Stage 4 output.

export AUTO_RESUME="${AUTO_RESUME:-1}"
export REBUILD_DATA="${REBUILD_DATA:-0}"
export RUN_NAME="${RUN_NAME:-conv_qwen3_t065_64k128k_balanced_v3}"
export STAGE3_STEPS="${STAGE3_STEPS:-9250}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XATTN_ROOT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn"
export STAGE3_INIT="${STAGE3_INIT:-${XATTN_ROOT}/qwen_weights/${RUN_NAME}/stage3_extend_96k128k_t065_s8_yarn4_bf16_ema_step9250.pt}"

echo "[combined] RUN_NAME=${RUN_NAME}"
echo "[combined] STAGE3_STEPS=${STAGE3_STEPS}"
echo "[combined] STAGE3_INIT=${STAGE3_INIT}"

# Preserve the original Stage 1-3 code and checkpoint layout, but stop before
# its legacy Stage 4/5 passes.
STOP_AFTER_STAGE3=1 \
  bash "${REPO_ROOT}/ft_scripts/run_ruler_mix_sparse_guarded_qwen.sh"

test -f "${STAGE3_INIT}" || {
  echo "[combined] required Stage-3 step-9250 EMA not found: ${STAGE3_INIT}" >&2
  echo "[combined] If the old run only has a different step, set STAGE3_INIT explicitly." >&2
  exit 1
}

# The corrective Stage 4 keeps the same experiment directory and resumes its
# own train_state if interrupted.
RUN_NAME="${RUN_NAME}" \
STAGE3_INIT="${STAGE3_INIT}" \
AUTO_RESUME="${AUTO_RESUME}" \
REBUILD_DATA="${REBUILD_DATA}" \
  bash "${REPO_ROOT}/ft_scripts/run_ruler_mix_sparse_guarded_qwen_stage4.sh"

echo "[combined] Stage 1-3 (through step 9250) and corrective Stage 4 complete."
