"""
Block-moment convolution sparse prefill.

Unlike Conv.py/XAttention, this estimator does not form a stride-sampled
token-by-token attention matrix and does not sum softmaxed small blocks.  It
works at the final sparse-attention granularity from the beginning:

    Q/K [B,H,N,D]
      -> moments over each exact 128-token block
      -> moment approximation of block attention mass [B,H,QB,KB]
      -> learned 7x7 per-layer/per-head convolution
      -> Top-P / Top-K block mask
      -> the existing exact 128x128 block-sparse attention

`Conv_prefill` is exported as a drop-in name so load_llama.py only needs its
import changed from xattn.src.Conv to xattn.src.Conv_ker.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

from block_sparse_attn import block_sparse_attn_func
from xattn.src.kernels_conv_ker import (
    block_group_moments,
    causal_block_softmax,
    conv2d_block_map,
)
from xattn.src.utils import find_blocks_chunked


_HARDCODED_WEIGHT_PATH = (
    "/inspire/hdd/global_user/gexinmu-253108100065/Repos/"
    "fuyicheng_workshop/Innovator-lm-evaluation-hardness/"
    "x-attention-main/xattn/conv_ker_weights_16k/"
    "conv_kernel_7x7_convker_moment_"
    "ruler_t07_16k_step10500.pt"
)

_WEIGHT_CACHE: dict[tuple[str, str], torch.Tensor] = {}
_STATIC_TENSOR_CACHE: dict[tuple[int, int, int, str], tuple[torch.Tensor, ...]] = {}


def _load_weight(device: torch.device, kernel_size: int = 7) -> torch.Tensor:
    # The path is intentionally hard-coded, matching the requested inference
    # deployment.  Edit this one constant when switching a checkpoint.
    path = _HARDCODED_WEIGHT_PATH
    key = (path, str(device))
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Conv_ker checkpoint does not exist: {path}. "
            "Edit _HARDCODED_WEIGHT_PATH in Conv_ker.py."
        )
    weight = torch.load(path, map_location=device, weights_only=True)
    if not torch.is_tensor(weight):
        raise TypeError(f"convolution checkpoint must be a Tensor, got {type(weight)}")
    if tuple(weight.shape[-2:]) != (kernel_size, kernel_size):
        raise ValueError(
            f"expected a {kernel_size}x{kernel_size} kernel, got {tuple(weight.shape)}"
        )
    weight = weight.detach().to(device=device, dtype=torch.float32).contiguous()
    _WEIGHT_CACHE[key] = weight
    return weight


def _weight_for_layer(
    weight: torch.Tensor,
    *,
    layer_idx: Optional[int],
    batch_size: int,
    num_heads: int,
    kernel_size: int,
) -> torch.Tensor:
    """Convert supported checkpoint layouts to [B*H,K,K]."""
    if weight.ndim == 2:
        selected = weight[None].expand(num_heads, -1, -1)
    elif weight.ndim == 3:
        if weight.shape[0] == 1:
            selected = weight.expand(num_heads, -1, -1)
        elif weight.shape[0] == num_heads:
            selected = weight
        else:
            raise ValueError(f"bad head dimension in weight {tuple(weight.shape)}")
    elif weight.ndim == 4:
        if weight.shape[:2] == (1, 1):
            selected = weight[0].expand(num_heads, -1, -1)
        elif weight.shape[0] == num_heads and weight.shape[1] == 1:
            selected = weight[:, 0]
        else:
            if layer_idx is None:
                raise ValueError(
                    f"layer_idx is required for weight {tuple(weight.shape)}"
                )
            selected = weight[layer_idx]
            if selected.shape[0] == 1:
                selected = selected.expand(num_heads, -1, -1)
            elif selected.shape[0] != num_heads:
                raise ValueError(
                    f"bad head dimension in weight {tuple(weight.shape)}"
                )
    elif weight.ndim == 5:
        if layer_idx is None:
            raise ValueError(f"layer_idx is required for weight {tuple(weight.shape)}")
        selected = weight[layer_idx]
        if selected.shape[0] == 1:
            selected = selected.expand(num_heads, -1, -1, -1)
        if selected.shape[0] != num_heads or selected.shape[1] != 1:
            raise ValueError(f"bad weight layout {tuple(weight.shape)}")
        selected = selected[:, 0]
    else:
        raise ValueError(f"unsupported weight layout {tuple(weight.shape)}")

    if tuple(selected.shape) != (num_heads, kernel_size, kernel_size):
        raise ValueError(f"selected kernel has bad shape {tuple(selected.shape)}")
    if batch_size > 1:
        selected = selected.repeat(batch_size, 1, 1)
    return selected.contiguous()


def _block_logits_from_moments(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    block_size: int,
    summary_groups: int,
    norm: float,
    variance_weight: float,
    phase_weight: float,
) -> torch.Tensor:
    """
    Approximate log E[exp(qk/sqrt(D))] with block moments.

    The second-order correction follows the Gaussian cumulant approximation:

        log E exp(X) ~= E[X] + 0.5 Var[X].

    A small coarse-phase term retains within-block positional structure without
    constructing any sampled token attention tiles.
    """
    q_mean_g, q_second_g = block_group_moments(
        query_states,
        block_size=block_size,
        summary_groups=summary_groups,
    )
    k_mean_g, k_second_g = block_group_moments(
        key_states,
        block_size=block_size,
        summary_groups=summary_groups,
    )

    q_mean = q_mean_g.mean(dim=-2)
    k_mean = k_mean_g.mean(dim=-2)
    q_second = q_second_g.mean(dim=-2)
    k_second = k_second_g.mean(dim=-2)

    head_dim = query_states.shape[-1]
    scale = 1.0 / (math.sqrt(head_dim) * float(norm))
    k_mean_t = k_mean.transpose(-1, -2)
    logits = torch.matmul(q_mean, k_mean_t).mul_(scale)

    if phase_weight:
        batch, heads, q_blocks, groups, dim = q_mean_g.shape
        k_blocks = k_mean_g.shape[2]
        q_phase = q_mean_g.reshape(batch, heads, q_blocks, groups * dim)
        k_phase = k_mean_g.reshape(batch, heads, k_blocks, groups * dim)
        phase_dot = torch.matmul(q_phase, k_phase.transpose(-1, -2))
        phase_dot.mul_(scale / float(groups))
        phase_dot.sub_(logits).mul_(float(phase_weight))
        logits.add_(phase_dot)
        del phase_dot

    del q_mean_g, k_mean_g, q_second_g, k_second_g

    # Var(q.k) = E[q²].E[k²] - E[q]².E[k]² under a dimension-wise
    # independence approximation. The in-place sequence limits peak block-map
    # memory at 64K/128K contexts.
    variance = torch.matmul(q_second, k_second.transpose(-1, -2))
    variance.sub_(
        torch.matmul(
            q_mean.square(),
            k_mean.square().transpose(-1, -2),
        )
    ).clamp_(min=0.0)
    variance.mul_(0.5 * float(variance_weight) * scale * scale).clamp_(max=8.0)
    logits.add_(variance)
    del variance

    return torch.nan_to_num_(
        logits.float(),
        nan=-1.0e4,
        posinf=1.0e4,
        neginf=-1.0e4,
    )


def _causal_geometry(q_blocks: int, k_blocks: int, device: torch.device):
    q_idx = torch.arange(q_blocks, device=device)
    k_idx = torch.arange(k_blocks, device=device)
    offset = k_blocks - q_blocks
    causal_map = k_idx[None, :] <= q_idx[:, None] + offset
    frontier = (q_idx + offset).clamp(0, k_blocks - 1)
    return q_idx, causal_map, frontier


def _select_mask(
    scores: torch.Tensor,
    *,
    threshold,
    causal: bool,
    keep_sink: bool,
    keep_recent: bool,
    fixed_topk: Optional[int],
    topk_ratio: Optional[float],
    fallback_topk: int,
) -> torch.Tensor:
    batch, heads, q_blocks, k_blocks = scores.shape
    q_idx, causal_map, frontier = _causal_geometry(
        q_blocks,
        k_blocks,
        scores.device,
    )
    energy = torch.nan_to_num(
        scores.float(),
        nan=-1.0e4,
        posinf=1.0e4,
        neginf=-1.0e4,
    )
    if causal:
        energy = energy.masked_fill(~causal_map[None, None], -1.0e9)

    if topk_ratio is not None:
        ratio = float(topk_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0,1], got {ratio}")
        visible = (
            q_idx + (k_blocks - q_blocks) + 1
            if causal
            else torch.full_like(q_idx, k_blocks)
        ).clamp(min=1, max=k_blocks)
        k_per_row = torch.ceil(visible.float() * ratio).long()
        # For causal prefill q_blocks <= k_blocks, the last query row can see
        # every key block. Keep this a Python integer to avoid a CUDA sync.
        max_visible = k_blocks
        max_k = int(math.ceil(max_visible * ratio))
        indices = torch.topk(energy, k=max(1, max_k), dim=-1).indices
        rank = torch.arange(indices.shape[-1], device=scores.device)
        sources = rank[None, None, None, :] < k_per_row[None, None, :, None]
        sources = sources.expand(batch, heads, -1, -1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, indices, sources)
    elif fixed_topk is not None:
        count = max(1, min(int(fixed_topk), k_blocks))
        indices = torch.topk(energy, k=count, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
    elif threshold is not None:
        # One full-map call has exactly the same current_index geometry that
        # chunked calls would have, but avoids Python work per long-text chunk.
        mask = find_blocks_chunked(
            scores,
            k_blocks - q_blocks,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )
    else:
        # Safe explicit fallback, useful for controlled ablations.
        count = max(1, min(int(fallback_topk), k_blocks))
        indices = torch.topk(energy, k=count, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, indices, True)

    if causal:
        mask &= causal_map[None, None]
        # Block-sparse causal attention requires the frontier block for every row.
        mask.scatter_(
            -1,
            frontier[None, None, :, None].expand(batch, heads, -1, 1),
            True,
        )
    if keep_sink:
        mask[..., 0] = True
    if keep_recent:
        mask.scatter_(
            -1,
            frontier[None, None, :, None].expand(batch, heads, -1, 1),
            True,
        )
    return mask.contiguous()


def conv_ker_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    block_size: int = 128,
    stride: int = 16,
    norm: float = 1.0,
    threshold=0.8,
    causal: bool = True,
    keep_sink: bool = False,
    keep_recent: bool = False,
    conv_kernel_size: int = 7,
    layer_idx: Optional[int] = None,
    fallback_topk: int = 8,
    fixed_topk: Optional[int] = None,
    topk_ratio: Optional[float] = None,
    summary_groups: int = 4,
    variance_weight: float = 0.5,
    phase_weight: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return unsmoothed block masses and the selected 128x128 sparse-block mask.

    `stride` is accepted for drop-in compatibility but is intentionally not
    used: all 128 tokens contribute to the moments. `summary_groups` replaces
    stride as the speed/positional-detail knob; 4 is the recommended default.
    """
    del stride
    if block_size != 128:
        raise ValueError("Conv_ker is intentionally fixed to block_size=128")
    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("Q and K must be [B,H,N,D]")
    if query_states.shape[:2] != key_states.shape[:2]:
        raise ValueError("Q and K batch/head dimensions must match")
    if query_states.shape[-1] != key_states.shape[-1]:
        raise ValueError("Q and K head dimensions must match")
    if causal and query_states.shape[2] > key_states.shape[2]:
        raise ValueError("causal prefill requires q_len <= k_len")
    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)

    batch, heads, _, _ = query_states.shape
    logits = _block_logits_from_moments(
        query_states,
        key_states,
        block_size=block_size,
        summary_groups=summary_groups,
        norm=norm,
        variance_weight=variance_weight,
        phase_weight=phase_weight,
    )
    block_mass = causal_block_softmax(
        logits,
        q_len=query_states.shape[2],
        k_len=key_states.shape[2],
        block_size=block_size,
        causal=causal,
    )

    raw_weight = _load_weight(query_states.device, conv_kernel_size)
    conv_weight = _weight_for_layer(
        raw_weight,
        layer_idx=layer_idx,
        batch_size=batch,
        num_heads=heads,
        kernel_size=conv_kernel_size,
    )
    smoothed = conv2d_block_map(block_mass, conv_weight)
    mask = _select_mask(
        smoothed,
        threshold=threshold,
        causal=causal,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        fixed_topk=fixed_topk,
        topk_ratio=topk_ratio,
        fallback_topk=fallback_topk,
    )
    return block_mass, mask


def _static_tensors(
    q_len: int,
    k_len: int,
    num_heads: int,
    device: torch.device,
):
    key = (q_len, k_len, num_heads, str(device))
    if key not in _STATIC_TENSOR_CACHE:
        _STATIC_TENSOR_CACHE[key] = (
            torch.tensor([0, q_len], dtype=torch.int32, device=device),
            torch.tensor([0, k_len], dtype=torch.int32, device=device),
            torch.ones(num_heads, dtype=torch.int32, device=device),
        )
    return _STATIC_TENSOR_CACHE[key]


def _density(mask: torch.Tensor, causal: bool) -> float:
    batch, heads, q_blocks, k_blocks = mask.shape
    if causal:
        offset = k_blocks - q_blocks
        valid = (
            torch.arange(q_blocks, device=mask.device, dtype=torch.float32)
            + float(offset + 1)
        ).clamp(min=0, max=k_blocks)
        denominator = valid.sum() * batch * heads
    else:
        denominator = torch.tensor(
            batch * heads * q_blocks * k_blocks,
            device=mask.device,
            dtype=torch.float32,
        )
    return float((mask.float().sum() / denominator.clamp_min(1)).cpu())


def ConvKer_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride: int,
    norm: float = 1.0,
    threshold=0.8,
    block_size: int = 128,
    use_triton: bool = True,
    causal: bool = True,
    kdb: int = 1,
    chunk_size=None,
    keep_sink: bool = False,
    keep_recent: bool = False,
    conv_kernel_size: int = 7,
    conv_weight_path=None,
    layer_idx: Optional[int] = None,
    conv_safe_topk: bool = False,
    fallback_topk: int = 8,
    return_density: bool = False,
    fixed_topk: Optional[int] = None,
    topk_ratio: Optional[float] = None,
    summary_groups: int = 4,
    variance_weight: float = 0.5,
    phase_weight: float = 0.15,
):
    """Drop-in sparse-prefill entry point using the block-moment estimator."""
    # Compatibility-only options from Conv.py. CPU/non-Triton fallback is
    # selected automatically inside kernels_conv_ker.
    del use_triton, kdb, chunk_size, conv_weight_path, conv_safe_topk

    batch, heads, q_len, head_dim = query_states.shape
    _, _, k_len, _ = key_states.shape
    if batch != 1:
        raise ValueError("block_sparse_attn_func currently requires batch_size=1")
    if block_size != 128:
        raise ValueError("block_sparse_attn_func requires block_size=128")

    block_mass, mask = conv_ker_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        causal=causal,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        conv_kernel_size=conv_kernel_size,
        layer_idx=layer_idx,
        fallback_topk=fallback_topk,
        fixed_topk=fixed_topk,
        topk_ratio=topk_ratio,
        summary_groups=summary_groups,
        variance_weight=variance_weight,
        phase_weight=phase_weight,
    )

    device = query_states.device
    if key_states.device != device:
        key_states = key_states.to(device)
    if value_states.device != device:
        value_states = value_states.to(device)
    q = query_states.transpose(1, 2).reshape(q_len, heads, head_dim)
    k = key_states.transpose(1, 2).reshape(k_len, heads, head_dim)
    v = value_states.transpose(1, 2).reshape(k_len, heads, head_dim)
    q_cu, k_cu, head_mask_type = _static_tensors(q_len, k_len, heads, device)

    output = block_sparse_attn_func(
        q,
        k,
        v,
        q_cu,
        k_cu,
        head_mask_type,
        None,
        mask,
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )
    output = output.view(batch, q_len, heads, head_dim).transpose(1, 2)
    if return_density:
        return output, _density(mask, causal)
    return output


# Drop-in import compatibility:
#   from xattn.src.Conv_ker import Conv_prefill
Conv_prefill = ConvKer_prefill
