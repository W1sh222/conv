"""Deterministic Conv-kernel initialisation used by the Llama baseline.

The baseline kernel is the one used by the original training code: a 7x7
vertical line and the main diagonal are set to one, with all other entries
set to zero.  Keeping the construction in code makes the baseline
reproducible even when no checkpoint file is present on the evaluation host.
"""

from __future__ import annotations

import torch


def make_vertical_plus_diag_kernel(
    size: int = 7,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a ``[size, size]`` vertical-plus-main-diagonal kernel."""
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"kernel size must be a positive odd integer, got {size}")
    weight = torch.zeros(size, size, dtype=dtype, device=device)
    center = size // 2
    weight[:, center] = 1.0
    idx = torch.arange(size, device=device)
    weight[idx, idx] = 1.0
    return weight


def make_layer_head_kernel(
    num_layers: int = 32,
    num_heads: int = 32,
    kernel_size: int = 7,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the old per-layer/per-head shape ``[L, H, K, K]``."""
    base = make_vertical_plus_diag_kernel(
        kernel_size, dtype=dtype, device=device
    )
    return base[None, None].repeat(num_layers, num_heads, 1, 1).contiguous()


def make_initial_conv_weight(
    kernel_size: int = 7,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a shared ``[1, 1, K, K]`` form accepted by ``Conv.py``."""
    return make_vertical_plus_diag_kernel(
        kernel_size, dtype=dtype, device=device
    )[None, None].contiguous()

