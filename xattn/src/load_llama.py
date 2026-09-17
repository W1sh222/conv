"""Transformers 4.51.0 adapter for Llama-3.1-8B-Instruct."""

from __future__ import annotations

from typing import Optional

from xattn.src.load_transformers_451 import (
    BaseFastPrefillConfig,
    _maybe_unpack_density,
    load_model_451,
)


LLAMA_NUM_LAYERS = 32
LLAMA_NUM_ATTENTION_HEADS = 32
LLAMA_NUM_KEY_VALUE_HEADS = 8
# Reproducible baseline: vertical line = 1 and main diagonal = 1.
# Conv.py expands this sentinel to [1, 1, 7, 7] at runtime; no learned
# checkpoint is loaded for this experiment.
LLAMA_CONV_WEIGHT_PATH = "initial_vertical_diag"


class FastPrefillConfig(BaseFastPrefillConfig):
    def __init__(self, threshold=None, stride=8, **kwargs):
        super().__init__(
            # A top-k ratio disables threshold selection in XAttention/Conv.
            # Keep a scalar fallback so the loader has no dependency on the
            # removed xattn.threshold package.
            threshold=0.9 if threshold is None else threshold,
            stride=stride,
            default_conv_weight_path=LLAMA_CONV_WEIGHT_PATH,
            **kwargs,
        )


def load_model(
    fastprefillconfig: Optional[FastPrefillConfig] = None,
    name_or_path: str = "",
):
    # Config attributes live outside the dict, so a supplied config can be falsy.
    config = FastPrefillConfig() if fastprefillconfig is None else fastprefillconfig
    return load_model_451(
        name_or_path=name_or_path,
        fastprefillconfig=config,
        expected_model_type="llama",
        expected_layers=LLAMA_NUM_LAYERS,
        expected_heads=LLAMA_NUM_ATTENTION_HEADS,
        expected_kv_heads=LLAMA_NUM_KEY_VALUE_HEADS,
    )


def load_fake_model(
    layer_to_save: int,
    target_len: int,
    name_or_path: str = "",
    output_dir: str = "output",
):
    """Compatibility helper used by the repository's efficiency benchmark."""
    model, tokenizer = load_model(
        FastPrefillConfig(metric="full"), name_or_path=name_or_path
    )
    for layer in model.model.layers:
        layer.self_attn.layer_to_save = int(layer_to_save)
        layer.self_attn.target_len = int(target_len)
        layer.self_attn.capture_output_dir = str(output_dir)
    return model, tokenizer
