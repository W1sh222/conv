"""Qwen2.5 adapter for the existing hard-coded Conv prefill implementation.

The project Conv implementation intentionally owns the checkpoint path in
``xattn.src.Conv``.  This module keeps that behavior unchanged and only adapts
legacy Llama-shaped per-layer/per-head kernels to Qwen2.5-7B's 28 layers and
28 query heads.

This makes the old checkpoint runnable on Qwen.  It does not turn a
Llama-trained checkpoint into a Qwen-trained checkpoint; for a scientifically
meaningful Conv comparison, replace the hard-coded checkpoint with one trained
for Qwen while keeping the same adapter interface.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from xattn.src import Conv as _conv


QWEN_NUM_LAYERS = 28
QWEN_NUM_ATTENTION_HEADS = 28
_WARNED_ABOUT_SLICING = False


def _adapt_hardcoded_weight(weight: torch.Tensor) -> torch.Tensor:
    """Slice only unambiguously layer/head-specific Llama kernels for Qwen."""
    global _WARNED_ABOUT_SLICING

    adapted = weight

    # [L, H, K, K]
    if weight.dim() == 4 and weight.shape[0] >= QWEN_NUM_LAYERS and weight.shape[1] > 1:
        adapted = weight[:QWEN_NUM_LAYERS, :QWEN_NUM_ATTENTION_HEADS]
    # [L, H, 1, K, K]
    elif weight.dim() == 5 and weight.shape[0] >= QWEN_NUM_LAYERS and weight.shape[1] > 1:
        adapted = weight[:QWEN_NUM_LAYERS, :QWEN_NUM_ATTENTION_HEADS]
    # [H, K, K] is unambiguously per-head.
    elif weight.dim() == 3 and weight.shape[0] > QWEN_NUM_ATTENTION_HEADS:
        adapted = weight[:QWEN_NUM_ATTENTION_HEADS]

    if adapted.shape != weight.shape and not _WARNED_ABOUT_SLICING:
        print(
            "[Conv-Qwen] adapting hard-coded Conv weight "
            f"from shape={tuple(weight.shape)} to shape={tuple(adapted.shape)}. "
            "The checkpoint is still the original hard-coded PT.",
            flush=True,
        )
        _WARNED_ABOUT_SLICING = True

    return adapted


@contextmanager
def _qwen_weight_adapter():
    """Temporarily adapt Conv's internally loaded hard-coded weight."""
    original_get_conv_weight = _conv._get_conv_weight

    def get_qwen_conv_weight(*args: Any, **kwargs: Any) -> torch.Tensor:
        return _adapt_hardcoded_weight(original_get_conv_weight(*args, **kwargs))

    _conv._get_conv_weight = get_qwen_conv_weight
    try:
        yield
    finally:
        _conv._get_conv_weight = original_get_conv_weight


def Conv_prefill(*args: Any, **kwargs: Any):
    """Run the existing Conv prefill with Qwen-compatible weight dimensions."""
    # Conv.py deliberately ignores conv_weight_path and uses its hard-coded PT.
    # Keep accepting the argument so callers share the same interface.
    with _qwen_weight_adapter():
        return _conv.Conv_prefill(*args, **kwargs)

