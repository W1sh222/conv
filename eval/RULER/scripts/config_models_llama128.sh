#!/usr/bin/env bash

# Llama-3.1-8B uses its native 128K RoPE configuration. Reuse the existing
# model/tokenizer registry and change only the fixed evaluation length.
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONFIG_DIR}/config_models.sh"

SEQ_LENGTHS=(131072)

# Do not apply Qwen3's YaRN settings to Llama 3.1. SDPA avoids allocating the
# quadratic 128K 4-D causal mask before the patched conv/xattn forward runs.
export ROPE_SCALING_TYPE="none"
export MAX_POSITION_EMBEDDINGS_OVERRIDE="131072"
export MODEL_ATTENTION_IMPLEMENTATION="sdpa"
