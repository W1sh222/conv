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
    "x-attention-main/xattn/conv_weights2/"
    "conv_kernel_7x7_ruler_mix_sparse_guarded_t060_boundary_corrective_"
    "24k40k_bf16_ema_step750.pt"
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
    config = fastprefillconfig or FastPrefillConfig()
    return load_model_451(
        name_or_path=name_or_path,
        fastprefillconfig=config,
        expected_model_type="qwen3",
        expected_layers=QWEN3_NUM_LAYERS,
        expected_heads=QWEN3_NUM_ATTENTION_HEADS,
        expected_kv_heads=QWEN3_NUM_KEY_VALUE_HEADS,
    )
