"""
Triton kernels for the block-moment convolution estimator.

This is deliberately not the XAttention-style sampled QK path.  It never
materializes a token/stride attention matrix.  The kernels operate directly on
the 128-token sparse-attention blocks:

  1. reduce every token block to first and second moments;
  2. normalize the resulting block logits with a causal block softmax;
  3. apply the learned per-head 2-D convolution on the block map.

The GEMMs between these stages are intentionally delegated to torch.matmul,
which routes the small dense block matrices to cuBLAS.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # Allows CPU-only tooling to import the module.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _block_group_moments_kernel(
        x_ptr,
        mean_ptr,
        second_ptr,
        stride_xb,
        stride_xh,
        stride_xn,
        stride_xd,
        stride_ob,
        stride_oh,
        stride_on,
        stride_og,
        stride_od,
        seq_len,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        group_size: tl.constexpr,
        num_groups: tl.constexpr,
        block_d: tl.constexpr,
    ):
        block_group = tl.program_id(0)
        bh = tl.program_id(1)
        d_tile = tl.program_id(2)

        block_id = block_group // num_groups
        group_id = block_group - block_id * num_groups
        batch_id = bh // num_heads
        head_id = bh - batch_id * num_heads

        token_start = block_id * block_size + group_id * group_size
        offs_t = tl.arange(0, group_size)
        offs_d = d_tile * block_d + tl.arange(0, block_d)
        token_idx = token_start + offs_t

        valid = (token_idx[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        x_offsets = (
            batch_id * stride_xb
            + head_id * stride_xh
            + token_idx[:, None] * stride_xn
            + offs_d[None, :] * stride_xd
        )
        x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)

        count = tl.minimum(group_size, tl.maximum(seq_len - token_start, 0))
        inv_count = 1.0 / tl.maximum(count.to(tl.float32), 1.0)
        mean = tl.sum(x, axis=0) * inv_count
        second = tl.sum(x * x, axis=0) * inv_count

        out_offsets = (
            batch_id * stride_ob
            + head_id * stride_oh
            + block_id * stride_on
            + group_id * stride_og
            + offs_d * stride_od
        )
        out_mask = offs_d < head_dim
        tl.store(mean_ptr + out_offsets, mean, mask=out_mask)
        tl.store(second_ptr + out_offsets, second, mask=out_mask)


    @triton.jit
    def _causal_block_softmax_kernel(
        logits_ptr,
        output_ptr,
        stride_lbh,
        stride_lq,
        stride_lk,
        stride_obh,
        stride_oq,
        stride_ok,
        q_blocks,
        k_blocks,
        q_to_k_offset,
        key_len,
        block_size: tl.constexpr,
        block_k: tl.constexpr,
        causal: tl.constexpr,
    ):
        row = tl.program_id(0)
        bh = tl.program_id(1)
        offs_k = tl.arange(0, block_k)

        valid = offs_k < k_blocks
        if causal:
            valid = valid & (offs_k <= row + q_to_k_offset)

        offsets = bh * stride_lbh + row * stride_lq + offs_k * stride_lk
        logits = tl.load(logits_ptr + offsets, mask=valid, other=-float("inf")).to(
            tl.float32
        )

        # A block represents a sum over keys, not one synthetic key.  The log
        # token count makes the block-level softmax agree with that measure and
        # also handles the final partial 128-token block.
        key_count = tl.minimum(
            block_size,
            tl.maximum(key_len - offs_k * block_size, 0),
        ).to(tl.float32)
        logits += tl.log(tl.maximum(key_count, 1.0))

        row_max = tl.max(logits, axis=0)
        numerator = tl.exp(logits - row_max)
        numerator = tl.where(valid, numerator, 0.0)
        denominator = tl.sum(numerator, axis=0)
        probs = numerator / tl.maximum(denominator, 1.0e-20)

        out_offsets = bh * stride_obh + row * stride_oq + offs_k * stride_ok
        tl.store(output_ptr + out_offsets, probs, mask=offs_k < k_blocks)


    @triton.jit
    def _conv2d_block_map_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        stride_xbh,
        stride_xq,
        stride_xk,
        stride_wbh,
        stride_wy,
        stride_wx,
        stride_obh,
        stride_oq,
        stride_ok,
        q_blocks,
        k_blocks,
        kernel_size: tl.constexpr,
        block_k: tl.constexpr,
    ):
        row = tl.program_id(0)
        bh = tl.program_id(1)
        k_tile = tl.program_id(2)
        offs_k = k_tile * block_k + tl.arange(0, block_k)
        pad = kernel_size // 2

        acc = tl.zeros([block_k], dtype=tl.float32)
        for ky in range(kernel_size):
            source_q = tl.maximum(0, tl.minimum(q_blocks - 1, row + ky - pad))
            for kx in range(kernel_size):
                source_k = tl.maximum(
                    0,
                    tl.minimum(k_blocks - 1, offs_k + kx - pad),
                )
                x_offsets = (
                    bh * stride_xbh
                    + source_q * stride_xq
                    + source_k * stride_xk
                )
                w_offset = bh * stride_wbh + ky * stride_wy + kx * stride_wx
                values = tl.load(
                    x_ptr + x_offsets,
                    mask=offs_k < k_blocks,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(weight_ptr + w_offset).to(tl.float32)
                acc += values * weight

        out_offsets = (
            bh * stride_obh + row * stride_oq + offs_k * stride_ok
        )
        tl.store(output_ptr + out_offsets, acc, mask=offs_k < k_blocks)


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def block_group_moments(
    states: torch.Tensor,
    *,
    block_size: int = 128,
    summary_groups: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E[x] and E[x²] for groups inside every 128-token block."""
    if states.ndim != 4:
        raise ValueError(f"states must be [B,H,N,D], got {tuple(states.shape)}")
    if block_size != 128:
        raise ValueError("Conv_ker uses the block-sparse attention block_size=128")
    if summary_groups <= 0 or block_size % summary_groups:
        raise ValueError(
            f"summary_groups must divide {block_size}, got {summary_groups}"
        )

    batch, heads, seq_len, head_dim = states.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    group_size = block_size // summary_groups

    # Keeping moments in fp32 avoids an unstable variance subtraction.  The
    # following block GEMMs may still use TF32 on supported NVIDIA GPUs.
    mean = torch.empty(
        batch,
        heads,
        num_blocks,
        summary_groups,
        head_dim,
        device=states.device,
        dtype=torch.float32,
    )
    second = torch.empty_like(mean)

    if states.is_cuda and _TRITON_AVAILABLE:
        block_d = min(32, _next_power_of_two(head_dim))
        grid = (
            num_blocks * summary_groups,
            batch * heads,
            triton.cdiv(head_dim, block_d),
        )
        _block_group_moments_kernel[grid](
            states,
            mean,
            second,
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            mean.stride(0),
            mean.stride(1),
            mean.stride(2),
            mean.stride(3),
            mean.stride(4),
            seq_len,
            num_heads=heads,
            head_dim=head_dim,
            block_size=block_size,
            group_size=group_size,
            num_groups=summary_groups,
            block_d=block_d,
        )
        return mean, second

    padded_len = num_blocks * block_size
    padded = torch.nn.functional.pad(
        states.float(),
        (0, 0, 0, padded_len - seq_len),
    )
    grouped = padded.view(
        batch,
        heads,
        num_blocks,
        summary_groups,
        group_size,
        head_dim,
    )
    valid = torch.arange(padded_len, device=states.device) < seq_len
    valid = valid.view(num_blocks, summary_groups, group_size)
    counts = valid.sum(-1).clamp_min(1).to(torch.float32)
    mean.copy_(grouped.sum(-2) / counts[None, None, :, :, None])
    second.copy_(
        grouped.square().sum(-2) / counts[None, None, :, :, None]
    )
    return mean, second


def causal_block_softmax(
    logits: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    block_size: int = 128,
    causal: bool = True,
) -> torch.Tensor:
    """Normalize block logits into non-negative attention-mass estimates."""
    if logits.ndim != 4:
        raise ValueError(f"logits must be [B,H,QB,KB], got {tuple(logits.shape)}")
    batch, heads, q_blocks, k_blocks = logits.shape
    output = torch.empty_like(logits, dtype=torch.float32)
    offset = k_blocks - q_blocks

    if logits.is_cuda and _TRITON_AVAILABLE:
        block_k = _next_power_of_two(k_blocks)
        if block_k > 65536:
            raise ValueError(f"k_blocks={k_blocks} is too large for row softmax")
        flat_in = logits.contiguous().view(batch * heads, q_blocks, k_blocks)
        flat_out = output.view(batch * heads, q_blocks, k_blocks)
        _causal_block_softmax_kernel[(q_blocks, batch * heads)](
            flat_in,
            flat_out,
            flat_in.stride(0),
            flat_in.stride(1),
            flat_in.stride(2),
            flat_out.stride(0),
            flat_out.stride(1),
            flat_out.stride(2),
            q_blocks,
            k_blocks,
            offset,
            k_len,
            block_size=block_size,
            block_k=block_k,
            causal=causal,
        )
        return output

    key_counts = (
        k_len
        - torch.arange(k_blocks, device=logits.device) * block_size
    ).clamp(min=0, max=block_size)
    adjusted = logits.float() + key_counts.clamp_min(1).float().log()
    if causal:
        q_idx = torch.arange(q_blocks, device=logits.device)[:, None]
        k_idx = torch.arange(k_blocks, device=logits.device)[None, :]
        adjusted = adjusted.masked_fill(
            k_idx > q_idx + offset,
            -torch.inf,
        )
    return torch.softmax(adjusted, dim=-1)


def conv2d_block_map(
    scores: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Apply per-(batch, head) KxK convolution with replicate padding.

    scores: [B,H,QB,KB], weight: [B*H,K,K].
    """
    if scores.ndim != 4:
        raise ValueError(f"scores must be [B,H,QB,KB], got {tuple(scores.shape)}")
    batch, heads, q_blocks, k_blocks = scores.shape
    if weight.ndim != 3 or weight.shape[0] != batch * heads:
        raise ValueError(
            f"weight must be [B*H,K,K], got {tuple(weight.shape)}"
        )
    kernel_size = int(weight.shape[-1])
    if weight.shape[-2] != kernel_size or kernel_size % 2 != 1:
        raise ValueError(f"kernel must be odd and square, got {tuple(weight.shape)}")

    x = scores.contiguous().float().view(batch * heads, q_blocks, k_blocks)
    w = weight.contiguous().float()
    output = torch.empty_like(x)

    if scores.is_cuda and _TRITON_AVAILABLE and kernel_size in (3, 5, 7, 9, 11):
        block_k = 64 if k_blocks >= 64 else _next_power_of_two(k_blocks)
        grid = (
            q_blocks,
            batch * heads,
            triton.cdiv(k_blocks, block_k),
        )
        _conv2d_block_map_kernel[grid](
            x,
            w,
            output,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            q_blocks,
            k_blocks,
            kernel_size=kernel_size,
            block_k=block_k,
        )
        return output.view(batch, heads, q_blocks, k_blocks)

    padded = torch.nn.functional.pad(
        x.unsqueeze(0),
        (kernel_size // 2,) * 4,
        mode="replicate",
    )
    result = torch.nn.functional.conv2d(
        padded,
        w[:, None, :, :],
        groups=batch * heads,
    )
    return result.view(batch, heads, q_blocks, k_blocks)

