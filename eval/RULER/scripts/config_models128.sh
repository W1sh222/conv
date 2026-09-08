#!/usr/bin/env bash

# Reuse the exact model/tokenizer registry from the normal RULER configuration,
# changing only the fixed evaluation length and Qwen3 long-context RoPE setup.
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONFIG_DIR}/config_models.sh"

SEQ_LENGTHS=(131072)

# Qwen3-8B is native below 128K.  These values match the static YaRN
# coordinates used by the 64K-128K Conv training pipeline.
export ROPE_SCALING_TYPE="${ROPE_SCALING_TYPE:-yarn}"
export ROPE_FACTOR="${ROPE_FACTOR:-4.0}"
export ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS="${ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS:-32768}"
export MAX_POSITION_EMBEDDINGS_OVERRIDE="${MAX_POSITION_EMBEDDINGS_OVERRIDE:-131072}"
