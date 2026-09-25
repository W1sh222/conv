"""Method adapters for a standalone Llama top-p/Conv speed benchmark.

The upstream XAttention benchmark compares methods on exactly the same
captured Q/K tensors.  These small adapters preserve that contract while
adding Conv.  In particular, XAttention and Conv intentionally do *not* pass
``topk_ratio``: the original top-p/threshold selector is used.
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from xattn.src.Conv import Conv_prefill
from xattn.src.Flexprefill import Flexprefill_prefill
from xattn.src.Fullprefill import Full_prefill
from xattn.src.Minference import Minference_prefill
from xattn.src.Xattention import Xattention_prefill


DEFAULT_THRESHOLD = 0.90
# Compromise between the previous high-quality/slow (0.97, 0.03) setting and
# the faster/original (0.95, 0.10) setting.
DEFAULT_FLEX_GAMMA = 0.96
DEFAULT_FLEX_TAU = 0.06
DEFAULT_CONV_WEIGHT = "initial_vertical_diag"


def _torch_full_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Memory-efficient dense baseline for installations without FlashInfer."""
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
    )


def call_method(
    method: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    stride: int,
    threshold: float,
    chunk_size: int,
    conv_weight_path: Optional[str],
    conv_layer: int,
    flex_gamma: float,
    flex_tau: float,
    full_backend: str,
    minference_vertical_size: int = 1000,
    minference_slash_size: int = 6096,
) -> torch.Tensor:
    """Run one method with a common top-p/128-token-block contract."""
    method = method.lower()
    if method == "full":
        if full_backend == "flashinfer":
            return Full_prefill(q, k, v, causal=True)
        if full_backend == "sdpa":
            return _torch_full_prefill(q, k, v)
        raise ValueError(f"unsupported full backend: {full_backend}")

    if method == "xattn":
        return Xattention_prefill(
            q,
            k,
            v,
            stride=stride,
            threshold=threshold,
            use_triton=True,
            chunk_size=chunk_size,
            causal=True,
        )

    if method == "conv":
        return Conv_prefill(
            q,
            k,
            v,
            stride=stride,
            threshold=threshold,
            use_triton=True,
            chunk_size=chunk_size,
            causal=True,
            conv_weight_path=conv_weight_path or DEFAULT_CONV_WEIGHT,
            layer_idx=conv_layer,
        )

    if method == "flex":
        # Flex expects [batch, sequence, heads, dim], unlike XAttention/Conv.
        # No topk_ratio is passed: this is the original gamma/tau path.
        return Flexprefill_prefill(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            gamma=flex_gamma,
            tau=flex_tau,
        )

    if method == "minference":
        return Minference_prefill(
            q,
            k,
            v,
            vertical_size=minference_vertical_size,
            slash_size=minference_slash_size,
        )

    raise ValueError(f"unsupported method: {method}")


def benchmark_prefill(
    method: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmups: int,
    iterations: int,
    stride: int,
    threshold: float,
    chunk_size: int,
    conv_weight_path: Optional[str],
    conv_layer: int,
    flex_gamma: float,
    flex_tau: float,
    full_backend: str,
    minference_vertical_size: int = 1000,
    minference_slash_size: int = 6096,
) -> float:
    """Warm up and time one method; synchronize around every timed call."""
    fn: Callable[[], torch.Tensor] = lambda: call_method(
        method,
        q,
        k,
        v,
        stride=stride,
        threshold=threshold,
        chunk_size=chunk_size,
        conv_weight_path=conv_weight_path,
        conv_layer=conv_layer,
        flex_gamma=flex_gamma,
        flex_tau=flex_tau,
        minference_vertical_size=minference_vertical_size,
        minference_slash_size=minference_slash_size,
        full_backend=full_backend,
    )

    for _ in range(warmups):
        out = fn()
        del out
    torch.cuda.synchronize(q.device)

    elapsed = 0.0
    for _ in range(iterations):
        torch.cuda.synchronize(q.device)
        started = time.perf_counter()
        out = fn()
        torch.cuda.synchronize(q.device)
        elapsed += time.perf_counter() - started
        del out
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed / float(iterations)


def estimate_density(
    method: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    stride: int,
    threshold: float,
    chunk_size: int,
    conv_weight_path: Optional[str],
    conv_layer: int,
    flex_gamma: float,
    flex_tau: float,
) -> Optional[float]:
    """Return optional kernel density without contaminating timing loops."""
    if method not in {"xattn", "conv"}:
        return None
    out = (
        Xattention_prefill(
            q,
            k,
            v,
            stride=stride,
            threshold=threshold,
            use_triton=True,
            chunk_size=chunk_size,
            causal=True,
            return_density=True,
        )
        if method == "xattn"
        else Conv_prefill(
            q,
            k,
            v,
            stride=stride,
            threshold=threshold,
            use_triton=True,
            chunk_size=chunk_size,
            causal=True,
            conv_weight_path=conv_weight_path or DEFAULT_CONV_WEIGHT,
            layer_idx=conv_layer,
            return_density=True,
        )
    )
    if isinstance(out, tuple) and len(out) == 2:
        density = out[1]
        if density is not None:
            # Current XAttention/Conv implementations may return either a
            # scalar Python float or a tensor, depending on the density
            # helper used by the selected kernel.
            if torch.is_tensor(density):
                return float(density.detach().float().mean().cpu().item())
            return float(density)
    return None
