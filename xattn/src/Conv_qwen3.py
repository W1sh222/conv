"""Qwen3-8B adapter for the project's hard-coded Conv prefill path."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
import os

import torch

from xattn.src import Conv as _conv


QWEN3_NUM_LAYERS = 36
QWEN3_NUM_ATTENTION_HEADS = 32
QWEN3_CONV_WEIGHT_PATH = (
    "/inspire/hdd/global_user/gexinmu-253108100065/Repos/"
    "fuyicheng_workshop/Innovator-lm-evaluation-hardness/"
    "x-attention-main/xattn/conv_qwen3/"
    "conv_kernel_7x7_qwen3_8b_ruler_mix_sparse_guarded_t065_"
    "multikey_qa2_48k64k_bf16_ema.pt"
)
_WEIGHT_CACHE = {}


def _get_qwen3_weight(kernel_size=7, weight_path=None, device=None):
    """Load the intentionally hard-coded Qwen3 checkpoint.

    ``weight_path`` is deliberately ignored to preserve the repository's
    hard-coded-PT convention requested for Conv inference.
    """
    del weight_path
    if kernel_size != 7:
        raise ValueError(f"Qwen3 Conv checkpoint is 7x7, got kernel_size={kernel_size}")
    if not os.path.isfile(QWEN3_CONV_WEIGHT_PATH):
        raise FileNotFoundError(
            "Qwen3 Conv PT not found. Train it first or update the hard-coded "
            f"QWEN3_CONV_WEIGHT_PATH in Conv_qwen3.py: {QWEN3_CONV_WEIGHT_PATH}"
        )
    cache_key = (QWEN3_CONV_WEIGHT_PATH, str(device))
    if cache_key not in _WEIGHT_CACHE:
        weight = torch.load(
            QWEN3_CONV_WEIGHT_PATH, map_location=device, weights_only=True
        )
        expected = (QWEN3_NUM_LAYERS, QWEN3_NUM_ATTENTION_HEADS, 7, 7)
        if not torch.is_tensor(weight) or tuple(weight.shape) != expected:
            actual = tuple(weight.shape) if torch.is_tensor(weight) else type(weight)
            raise ValueError(f"Qwen3 Conv PT must have shape {expected}, got {actual}")
        if not torch.isfinite(weight).all():
            raise ValueError("Qwen3 Conv PT contains NaN or Inf")
        _WEIGHT_CACHE[cache_key] = weight
        print(f"[Conv-Qwen3] loaded hard-coded PT: {QWEN3_CONV_WEIGHT_PATH}", flush=True)
    return _WEIGHT_CACHE[cache_key]


@contextmanager
def _qwen3_weight_loader():
    original = _conv._get_conv_weight
    _conv._get_conv_weight = _get_qwen3_weight
    try:
        yield
    finally:
        _conv._get_conv_weight = original


def Conv_prefill(*args: Any, **kwargs: Any):
    with _qwen3_weight_loader():
        return _conv.Conv_prefill(*args, **kwargs)
