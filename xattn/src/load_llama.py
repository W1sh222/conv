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
LLAMA_CONV_WEIGHT_PATH = (
    "/inspire/hdd/global_user/gexinmu-253108100065/Repos/"
    "fuyicheng_workshop/Innovator-lm-evaluation-hardness/"
    "x-attention-main/xattn/conv_weights2/"
    "conv_kernel_7x7_ruler_mix_sparse_guarded_t065_multikey_qa2_"
    "48k64k_bf16_ema_step11000.pt"
)


class FastPrefillConfig(BaseFastPrefillConfig):
    def __init__(self, threshold=None, stride=16, **kwargs):
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
    config = fastprefillconfig or FastPrefillConfig()
    return load_model_451(
        name_or_path=name_or_path,
        fastprefillconfig=config,
        expected_model_type="llama",
        expected_layers=LLAMA_NUM_LAYERS,
        expected_heads=LLAMA_NUM_ATTENTION_HEADS,
        expected_kv_heads=LLAMA_NUM_KEY_VALUE_HEADS,
    )


def load_fake_model(layer_to_save: int, target_len: int, name_or_path: str = ""):
    """Compatibility helper used by the repository's efficiency benchmark."""
    model, tokenizer = load_model(
        FastPrefillConfig(metric="full"), name_or_path=name_or_path
    )
    for layer in model.model.layers:
        layer.self_attn.layer_to_save = int(layer_to_save)
        layer.self_attn.target_len = int(target_len)
    return model, tokenizer
