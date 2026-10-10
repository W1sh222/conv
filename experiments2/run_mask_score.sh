#!/usr/bin/env bash
# Reuse the exact sample/settings underlying the layer-14/head-8 mask.
set -euo pipefail
SOURCE_RUN=${1:-output/ruler_observation/vt_32k_seed42_q220_layers/layer_14/swap}
OUTPUT=${2:-output/ruler_observation/mask_score_layer14_head8}
shift $(( $# >= 2 ? 2 : $# ))
python -u experiments2/run_mask_score.py --source-run "$SOURCE_RUN" --output "$OUTPUT" "$@"
