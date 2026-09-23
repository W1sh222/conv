"""Transformers 4.51.0 adapter for Qwen3-8B."""

from __future__ import annotations

from typing import Optional

from xattn.src.load_transformers_451 import (
    BaseFastPrefillConfig,
    _maybe_unpack_density,
    load_model_451,
)


QWEN3_NUM_LAYERS = 36
QWEN3_NUM_ATTENTION_HEADS = 32
QWEN3_NUM_KEY_VALUE_HEADS = 8
QWEN3_CONV_WEIGHT_PATH = (
    "/inspire/hdd/global_user/gexinmu-253108100065/Repos/"
    "fuyicheng_workshop/Innovator-lm-evaluation-hardness/"
    "x-attention-main/xattn/qwen_weights/"
    "conv_qwen3_t065_64k128k_balanced_v3/"
    "stage3_extend_96k128k_t065_s8_yarn4_bf16_ema_step9250.pt"
)


class FastPrefillConfig(BaseFastPrefillConfig):
    def __init__(self, threshold=None, **kwargs):
        super().__init__(
            threshold=0.9 if threshold is None else threshold,
            default_conv_weight_path=QWEN3_CONV_WEIGHT_PATH,
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
        expected_model_type="qwen3",
        expected_layers=QWEN3_NUM_LAYERS,
        expected_heads=QWEN3_NUM_ATTENTION_HEADS,
        expected_kv_heads=QWEN3_NUM_KEY_VALUE_HEADS,
    )


def load_fake_model(
    layer_to_save: int,
    target_len: int,
    name_or_path: str = "",
    output_dir: str = "output",
):
    """Load Qwen3 for the efficiency benchmark and capture one layer's Q/K.

    The benchmark uses the same capture protocol for Llama and Qwen3.  This
    compatibility helper intentionally loads the dense/full path while the
    model is used only to build the cached query/key tensors; the subsequent
    timing section benchmarks each selected prefill implementation directly.
    """
    model, tokenizer = load_model(
        FastPrefillConfig(metric="full"), name_or_path=name_or_path
    )
    for layer in model.model.layers:
        layer.self_attn.layer_to_save = int(layer_to_save)
        layer.self_attn.target_len = int(target_len)
        layer.self_attn.capture_output_dir = str(output_dir)
    return model, tokenizer
