import math
import inspect
import torch
import torch.nn.functional as F

try:
    from xattn.src.utils import find_blocks_chunked
except Exception:
    find_blocks_chunked = None


def repeat_kv_to_q_heads(x: torch.Tensor, q_heads: int) -> torch.Tensor:
    b, kv_heads, s, d = x.shape

    if kv_heads == q_heads:
        return x

    if q_heads % kv_heads != 0:
        raise ValueError(f"Cannot repeat kv_heads={kv_heads} to q_heads={q_heads}")

    groups = q_heads // kv_heads
    return (
        x[:, :, None, :, :]
        .expand(b, kv_heads, groups, s, d)
        .reshape(b, q_heads, s, d)
    )


def build_causal_mask(seq_len: int, device) -> torch.Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


def dense_attention_teacher(q, k, v):
    """
    q/k/v: [1, heads, seq, head_dim]
    """
    _, _, seq_len, head_dim = q.shape

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = build_causal_mask(seq_len, q.device)

    scores = scores.masked_fill(
        causal_mask[None, None, :, :],
        float("-inf"),
    )

    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v)

    return out, probs, scores


def probs_to_block_scores(probs, block_size: int = 128):
    """
    probs: [1, heads, seq, seq]

    return:
        block_scores: [1, heads, q_blocks, k_blocks]

    每个 block_scores[..., i, j] 表示：
        query block i 对 key block j 的 dense attention 概率质量总和。
    """
    b, h, q_len, k_len = probs.shape

    q_pad = (block_size - q_len % block_size) % block_size
    k_pad = (block_size - k_len % block_size) % block_size

    if q_pad or k_pad:
        probs = F.pad(probs, (0, k_pad, 0, q_pad), value=0)

    q_total = q_len + q_pad
    k_total = k_len + k_pad

    q_blocks = q_total // block_size
    k_blocks = k_total // block_size

    block_scores = (
        probs.reshape(b, h, q_blocks, block_size, k_blocks, block_size)
        .sum(dim=-1)
        .sum(dim=-2)
    )

    return block_scores




# ----------------------------- kernels_conv block scorer -----------------------------

try:
    # Conv.py uses the full inverse anti-diagonal estimator.  Training must use
    # the same implementation; xattn.src.kernels_conv is the spaced-sampling
    # approximation and produces a different score map and scale.
    from xattn.src.kernels_conv_no_spaced_sampling import (
        flat_group_gemm_fuse_reshape as _kc_flat_group_gemm_fuse_reshape,
        softmax_fuse_block_sum as _kc_softmax_fuse_block_sum,
    )
    _KERNELS_CONV_AVAILABLE = True
except Exception:
    _kc_flat_group_gemm_fuse_reshape = None
    _kc_softmax_fuse_block_sum = None
    _KERNELS_CONV_AVAILABLE = False


def _ceil_to_multiple(x: int, m: int) -> int:
    return ((int(x) + int(m) - 1) // int(m)) * int(m)


def choose_conv_prefill_chunk_size(seq_len: int) -> int:
    """Use the same automatic chunk-size rule as xattn.src.Conv.Conv_prefill."""
    power_of_two = 1 << (int(seq_len) - 1).bit_length()
    return int(
        max(
            min(
                max(2048, power_of_two),
                128 * 1024 * 2048 // power_of_two,
            ),
            2048,
        )
    )


def _choose_kernels_conv_chunk_size(seq_len: int, stride: int, block_size: int, chunk_size: int | None) -> int:
    """
    kernels_conv.flat_group_gemm_fuse_reshape requires padded length to be
    divisible by stride * 128 because it keeps BLOCK_M=BLOCK_N=128.
    """
    unit = int(stride) * 128
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = choose_conv_prefill_chunk_size(seq_len)
    chunk_size = max(int(chunk_size), unit)
    chunk_size = _ceil_to_multiple(chunk_size, unit)
    return _ceil_to_multiple(seq_len, chunk_size)


def _choose_softmax_segment_size(k_reshaped_seq_len: int, reshaped_block_size: int) -> int:
    """
    Keep the same mathematical result as softmax_fuse_block_sum while allowing
    a larger segment. Default 128; can be overridden by TRAIN_KERNELS_CONV_SEGMENT_SIZE.
    """
    target = int(__import__('os').environ.get('TRAIN_KERNELS_CONV_SEGMENT_SIZE', '128'))
    target = max(int(reshaped_block_size), min(target, int(k_reshaped_seq_len)))
    seg = (target // int(reshaped_block_size)) * int(reshaped_block_size)
    while seg >= int(reshaped_block_size):
        if int(k_reshaped_seq_len) % seg == 0:
            return int(seg)
        seg -= int(reshaped_block_size)
    return int(reshaped_block_size)


def kernels_conv_block_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int = 128,
    stride: int = 16,
    norm: float = 1.0,
    chunk_size: int | None = 16384,
    sample_kernel_size: int = 7,
    causal: bool = True,
) -> torch.Tensor:
    """
    Estimate block scores with the same no-spaced-sampling kernel as Conv.py
    instead of using full dense attention probabilities.

    q, k: [B, H, seq, D], with K already repeated to Q heads.

    Return:
        block_scores: [B, H, q_blocks, k_blocks]

    Note:
        This changes only the block-score source. The training script can still
        use dense_out / dense_scores as the teacher for output and surrogate loss.
    """
    if not _KERNELS_CONV_AVAILABLE or _kc_flat_group_gemm_fuse_reshape is None:
        raise ImportError(
            'xattn.src.kernels_conv is required for kernels_conv_block_scores, '
            'but it could not be imported.'
        )

    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(f'Expected q/k shape [B,H,S,D], got q={tuple(q.shape)}, k={tuple(k.shape)}')
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(f'q/k shape mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}')

    batch_size, num_heads, q_len, head_dim = q.shape
    _, _, k_len, _ = k.shape
    device = q.device

    padded_len = _choose_kernels_conv_chunk_size(max(q_len, k_len), stride, block_size, chunk_size)
    q_pad = padded_len - q_len
    k_pad = padded_len - k_len

    if q_pad > 0:
        pad_q = F.pad(q, (0, 0, 0, q_pad), value=0)
    else:
        pad_q = q
    if k_pad > 0:
        pad_k = F.pad(k, (0, 0, 0, k_pad), value=0)
    else:
        pad_k = k

    pad_q = pad_q.contiguous()
    pad_k = pad_k.contiguous()

    q_real_block_num = (q_len + int(block_size) - 1) // int(block_size)
    k_real_block_num = (k_len + int(block_size) - 1) // int(block_size)
    reshaped_block_size = int(block_size) // int(stride)
    reshaped_chunk_size = int(chunk_size or padded_len) // int(stride)
    # If chunk_size was rounded up, use the effective padded chunk for iteration.
    effective_chunk_size = min(int(chunk_size or padded_len), padded_len)
    effective_chunk_size = _ceil_to_multiple(max(effective_chunk_size, int(stride) * 128), int(stride) * 128)
    effective_chunk_size = min(effective_chunk_size, padded_len)
    reshaped_chunk_size = effective_chunk_size // int(stride)
    q_chunk_num = padded_len // effective_chunk_size
    k_reshaped_seq_len = padded_len // int(stride)
    # Same convention as inference: real compressed length is ceil(real_len / stride).
    k_reshaped_real_len = k_reshaped_seq_len - (k_pad // int(stride))

    has_sample_arg = 'sample_kernel_size' in inspect.signature(_kc_flat_group_gemm_fuse_reshape).parameters
    diag_sample_kernel_size = max(1, min(int(sample_kernel_size), int(stride)))
    # The no-spaced-sampling kernel evaluates all `stride` offsets and Conv.py
    # consequently divides logits by the full stride.  sample_kernel_size is
    # accepted by that kernel only for API compatibility and is ignored.
    sampled_stride_norm = float(stride)

    softmax_segment_size = _choose_softmax_segment_size(
        k_reshaped_seq_len=k_reshaped_seq_len,
        reshaped_block_size=reshaped_block_size,
    )

    attn_sum_list = []
    for chunk_idx in range(q_chunk_num):
        q_start = chunk_idx * effective_chunk_size
        q_end = q_start + effective_chunk_size
        chunk_q = pad_q[:, :, q_start:q_end, :]

        chunk_start = chunk_idx * reshaped_chunk_size
        chunk_end = chunk_start + reshaped_chunk_size

        if has_sample_arg:
            attn_weights_slice = _kc_flat_group_gemm_fuse_reshape(
                chunk_q,
                pad_k,
                int(stride),
                chunk_start,
                chunk_end,
                is_causal=causal,
                sample_kernel_size=diag_sample_kernel_size,
            )
        else:
            attn_weights_slice = _kc_flat_group_gemm_fuse_reshape(
                chunk_q,
                pad_k,
                int(stride),
                chunk_start,
                chunk_end,
                is_causal=causal,
            )

        attn_sum = _kc_softmax_fuse_block_sum(
            attn_weights_slice,
            reshaped_block_size,
            softmax_segment_size,
            chunk_start,
            chunk_end,
            k_reshaped_real_len,
            1.4426950408889634 / math.sqrt(head_dim) / sampled_stride_norm / float(norm),
            is_causal=causal,
        )
        attn_sum_list.append(attn_sum)
        del attn_weights_slice

    if len(attn_sum_list) == 1:
        block_scores = attn_sum_list[0]
    else:
        block_scores = torch.cat(attn_sum_list, dim=-2)

    return block_scores[:, :, :q_real_block_num, :k_real_block_num].contiguous()



def kernels_conv_block_scores_infer_full(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int = 128,
    stride: int = 16,
    norm: float = 1.0,
    chunk_size: int | None = 0,
    sample_kernel_size: int = 7,
    causal: bool = True,
):
    """
    Inference-matched full block-score estimator for training.

    This follows the same prefill estimator path used by Conv_prefill:
      1) pad Q/K to chunk_size,
      2) call xattn.src.kernels_conv_no_spaced_sampling exactly as Conv.py,
      3) call softmax_fuse_block_sum,
      4) return the FULL padded block map.

    The caller should apply the trainable 7x7 conv on the full padded map, then
    use make_inference_chunked_block_mask(...), and only then crop real q/k
    blocks for loss/statistics.

    q/k: [B, H, seq, D], with K already repeated to Q heads.

    Returns:
      block_scores_full: [B,H,q_full_blocks,k_full_blocks]
      meta: dict with q_real_blocks, k_real_blocks, q_full_blocks,
            k_full_blocks, num_blocks_per_chunk.
    """
    if not _KERNELS_CONV_AVAILABLE or _kc_flat_group_gemm_fuse_reshape is None:
        raise ImportError(
            "xattn.src.kernels_conv is required for inference-matched sampled training, "
            "but it could not be imported."
        )
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(f"Expected q/k shape [B,H,S,D], got q={tuple(q.shape)}, k={tuple(k.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(f"q/k shape mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}")

    b, h, q_len, head_dim = q.shape
    _, _, k_len, _ = k.shape

    unit = int(stride) * 128
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = choose_conv_prefill_chunk_size(max(q_len, k_len))
    chunk_size = max(int(chunk_size), unit)
    chunk_size = _ceil_to_multiple(chunk_size, unit)

    q_pad = _ceil_to_multiple(q_len, chunk_size) - q_len
    k_pad = _ceil_to_multiple(k_len, chunk_size) - k_len

    pad_q = F.pad(q, (0, 0, 0, q_pad), value=0) if q_pad > 0 else q
    pad_k = F.pad(k, (0, 0, 0, k_pad), value=0) if k_pad > 0 else k
    pad_q = pad_q.contiguous()
    pad_k = pad_k.contiguous()

    q_padded_len = q_len + q_pad
    k_padded_len = k_len + k_pad

    q_full_blocks = q_padded_len // int(block_size)
    k_full_blocks = k_padded_len // int(block_size)
    q_real_blocks = (q_len + int(block_size) - 1) // int(block_size)
    k_real_blocks = (k_len + int(block_size) - 1) // int(block_size)

    reshaped_chunk_size = int(chunk_size) // int(stride)
    reshaped_block_size = int(block_size) // int(stride)
    num_blocks_per_chunk = int(chunk_size) // int(block_size)

    q_chunk_num = q_padded_len // int(chunk_size)
    k_chunk_num = k_padded_len // int(chunk_size)
    if k_chunk_num < q_chunk_num:
        raise ValueError(f"Expected k_chunk_num >= q_chunk_num, got {k_chunk_num} < {q_chunk_num}")

    k_reshaped_seq_len = k_padded_len // int(stride)
    k_reshaped_num_to_pad = k_pad // int(stride)
    k_reshaped_real_len = k_reshaped_seq_len - k_reshaped_num_to_pad

    has_sample_arg = "sample_kernel_size" in inspect.signature(_kc_flat_group_gemm_fuse_reshape).parameters
    diag_sample_kernel_size = max(1, min(int(sample_kernel_size), int(stride)))
    # Conv.py's no-spaced estimator always uses all offsets and divides by the
    # full stride.  Its sample_kernel_size argument is intentionally ignored.
    sampled_stride_norm = float(stride)

    softmax_segment_size = _choose_softmax_segment_size(
        k_reshaped_seq_len=k_reshaped_seq_len,
        reshaped_block_size=reshaped_block_size,
    )

    attn_sum_list = []
    for chunk_idx in range(q_chunk_num):
        q_start = chunk_idx * int(chunk_size)
        q_end = q_start + int(chunk_size)
        chunk_q = pad_q[:, :, q_start:q_end, :]

        # Same offset convention as Conv_prefill/conv_estimate.
        chunk_start = (
            (k_full_blocks - q_full_blocks) * reshaped_block_size
            + chunk_idx * reshaped_chunk_size
        )
        chunk_end = chunk_start + reshaped_chunk_size

        if has_sample_arg:
            attn_weights_slice = _kc_flat_group_gemm_fuse_reshape(
                chunk_q,
                pad_k,
                int(stride),
                int(chunk_start),
                int(chunk_end),
                is_causal=causal,
                sample_kernel_size=diag_sample_kernel_size,
            )
        else:
            attn_weights_slice = _kc_flat_group_gemm_fuse_reshape(
                chunk_q,
                pad_k,
                int(stride),
                int(chunk_start),
                int(chunk_end),
                is_causal=causal,
            )

        attn_sum = _kc_softmax_fuse_block_sum(
            attn_weights_slice,
            reshaped_block_size,
            softmax_segment_size,
            int(chunk_start),
            int(chunk_end),
            int(k_reshaped_real_len),
            1.4426950408889634 / math.sqrt(head_dim) / sampled_stride_norm / float(norm),
            is_causal=causal,
        )
        attn_sum_list.append(attn_sum)
        del attn_weights_slice

    if len(attn_sum_list) == 1:
        block_scores_full = attn_sum_list[0]
    else:
        block_scores_full = torch.cat(attn_sum_list, dim=-2)

    meta = {
        "q_real_blocks": int(q_real_blocks),
        "k_real_blocks": int(k_real_blocks),
        "q_full_blocks": int(q_full_blocks),
        "k_full_blocks": int(k_full_blocks),
        "num_blocks_per_chunk": int(num_blocks_per_chunk),
    }
    return block_scores_full.contiguous(), meta


def make_inference_chunked_block_mask(
    energy_full: torch.Tensor,
    threshold: float,
    q_real_blocks: int,
    k_real_blocks: int,
    num_blocks_per_chunk: int,
    causal: bool = True,
    topk_ratio: float | None = None,
):
    """
    Match Conv_prefill block selection during training.

    Inference uses either its fixed causal-visible top-k ratio selector or
    find_blocks_chunked on each query chunk of the FULL padded smoothed score
    map, then crops to real q/k blocks. `topk_ratio` takes priority over the
    cumulative-score `threshold`, matching Conv.py.
    """
    if find_blocks_chunked is None:
        raise ImportError("find_blocks_chunked is required for inference-matched training.")

    b, h, q_full_blocks, k_full_blocks = energy_full.shape
    device = energy_full.device
    energy_full = torch.nan_to_num(
        energy_full.float(),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    num_blocks_per_chunk = max(1, int(num_blocks_per_chunk))
    q_chunk_num = (q_full_blocks + num_blocks_per_chunk - 1) // num_blocks_per_chunk
    simple_mask_list = []
    for chunk_idx in range(q_chunk_num):
        chunk_start = chunk_idx * num_blocks_per_chunk
        chunk_end = min((chunk_idx + 1) * num_blocks_per_chunk, q_full_blocks)
        score_chunk = energy_full[:, :, chunk_start:chunk_end, :]
        offset = (
            k_full_blocks
            - q_full_blocks
            + chunk_idx * num_blocks_per_chunk
        )
        if topk_ratio is not None:
            ratio = float(topk_ratio)
            if not 0.0 < ratio <= 1.0:
                raise ValueError(
                    f"topk_ratio must be in (0,1], got {topk_ratio}"
                )
            batch, heads, chunk_rows, key_blocks = score_chunk.shape
            q_idx = torch.arange(chunk_rows, device=device)
            k_idx = torch.arange(key_blocks, device=device)
            if causal:
                valid = k_idx[None, :] <= q_idx[:, None] + int(offset)
                visible = (
                    q_idx + int(offset) + 1
                ).clamp(min=0, max=key_blocks)
            else:
                valid = torch.ones(
                    chunk_rows,
                    key_blocks,
                    dtype=torch.bool,
                    device=device,
                )
                visible = torch.full(
                    (chunk_rows,),
                    key_blocks,
                    dtype=torch.long,
                    device=device,
                )
            masked = score_chunk.masked_fill(
                ~valid[None, None],
                -1.0e9,
            )
            keep_per_row = torch.ceil(visible.float() * ratio).long()
            max_keep = max(
                1,
                int(math.ceil(key_blocks * ratio)),
            )
            indices = torch.topk(
                masked,
                k=min(max_keep, key_blocks),
                dim=-1,
            ).indices
            ranks = torch.arange(indices.shape[-1], device=device)
            sources = (
                ranks[None, None, None, :]
                < keep_per_row[None, None, :, None]
            ).expand(batch, heads, -1, -1)
            simple_mask = torch.zeros_like(
                score_chunk,
                dtype=torch.bool,
            )
            simple_mask.scatter_(-1, indices, sources)
            simple_mask &= valid[None, None]
        else:
            simple_mask = find_blocks_chunked(
                score_chunk,
                offset,
                threshold,
                None,
                decoding=False,
                mode="prefill",
                causal=causal,
            )
        simple_mask_list.append(simple_mask)

    mask_full = torch.cat(simple_mask_list, dim=-2)

    if causal:
        offset_full = k_full_blocks - q_full_blocks
        causal_full = torch.tril(
            torch.ones(q_full_blocks, k_full_blocks, dtype=torch.bool, device=device),
            diagonal=offset_full,
        )
        mask_full = mask_full & causal_full[None, None, :, :]

    # Same effective crop as Conv_prefill's _sanitize_block_sparse_mask:
    # use the leading real q/k blocks after padding.
    mask = mask_full[:, :, :q_real_blocks, :k_real_blocks].contiguous()

    # Safety: at least keep current/diagonal block for each real q block.
    offset = k_real_blocks - q_real_blocks
    diag_like = torch.zeros(q_real_blocks, k_real_blocks, dtype=torch.bool, device=device)
    for i in range(q_real_blocks):
        j = i + offset
        if 0 <= j < k_real_blocks:
            diag_like[i, j] = True
    mask = mask | diag_like[None, None, :, :]

    if causal:
        causal_real = torch.tril(
            torch.ones(q_real_blocks, k_real_blocks, dtype=torch.bool, device=device),
            diagonal=offset,
        )
        mask = mask & causal_real[None, None, :, :]

    return mask.contiguous()


def normalize_layer_head_weight(conv_weight, num_heads: int):
    """
    训练侧每一步传进来的 conv_weight 推荐 shape:

        [heads, 7, 7]

    兼容：
        [heads, 7, 7]
        [heads, 1, 7, 7]
        [1, 1, 7, 7]
        [7, 7]

    返回 grouped conv2d 用的：
        [heads, 1, 7, 7]
    """
    if conv_weight.dim() == 2:
        conv_weight = conv_weight[None, None, :, :].expand(
            num_heads, 1, 7, 7
        )

    elif conv_weight.dim() == 3:
        # [H, 7, 7]
        if conv_weight.shape[0] != num_heads:
            raise ValueError(
                f"conv_weight heads={conv_weight.shape[0]} != num_heads={num_heads}"
            )
        conv_weight = conv_weight[:, None, :, :]

    elif conv_weight.dim() == 4:
        # [H, 1, 7, 7] or [1, 1, 7, 7]
        if conv_weight.shape[0] == 1:
            conv_weight = conv_weight.expand(num_heads, 1, 7, 7)
        elif conv_weight.shape[0] != num_heads:
            raise ValueError(
                f"conv_weight shape {tuple(conv_weight.shape)} "
                f"incompatible with num_heads={num_heads}"
            )

    else:
        raise ValueError(f"Unsupported conv_weight shape: {tuple(conv_weight.shape)}")

    return conv_weight.contiguous()


def apply_conv_energy(block_scores, conv_weight):
    """
    block_scores: [1, heads, q_blocks, k_blocks]
    conv_weight:  [heads, 7, 7]

    return:
        energy: [1, heads, q_blocks, k_blocks]
    """
    b, h, qb, kb = block_scores.shape

    x = block_scores.float().reshape(1, b * h, qb, kb)

    weight = normalize_layer_head_weight(conv_weight, h)
    weight = weight.to(device=x.device, dtype=x.dtype)

    if b != 1:
        weight = weight.repeat(b, 1, 1, 1)

    pad = 3
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")

    energy = F.conv2d(
        x,
        weight,
        stride=1,
        padding=0,
        groups=b * h,
    )

    return energy.reshape(b, h, qb, kb)


def make_causal_block_mask(qb: int, kb: int, device) -> torch.Tensor:
    """
    构造 block-level causal mask:

        shape: [q_blocks, k_blocks]

    True 表示该 key block 对 query block 可见。
    """
    offset = kb - qb
    causal_block = torch.tril(
        torch.ones(qb, kb, dtype=torch.bool, device=device),
        diagonal=offset,
    )
    return causal_block


def make_hard_block_mask(
    energy,
    threshold: float = 0.8,
    block_size: int = 128,
    fallback_topk: int = 8,
    selector_mode: str = "topp",
    min_topk: int = 1,
    max_topk: int | None = None,
    exact_topk: int | None = None,
):
    """
    Correct block selector used by training.

    energy: [B, H, q_blocks, k_blocks]

    selector_mode:
      - "topp": true cumulative-probability top-p over causal key blocks.
          `threshold` is cumulative probability mass, so larger threshold keeps
          MORE blocks. `max_topk` can cap the number selected per query block.
      - "fixed_topk": keep exactly `exact_topk` if provided, else `max_topk`,
          else `fallback_topk` blocks per query/head before causal/diagonal guard.

    Notes:
      1. This replaces the old pseudo-top-p rule
            topk = max(fallback_topk, ceil(k_blocks * (1 - threshold)))
         because that made selected block count mostly a hand-coded function of
         threshold rather than something learned from energy sharpness.
      2. Diagonal/near-current block is always kept as a safety guard.
    """
    b, h, qb, kb = energy.shape
    device = energy.device

    energy = torch.nan_to_num(
        energy.float(),
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    )

    causal_block = make_causal_block_mask(qb, kb, device)
    masked_energy = energy.masked_fill(~causal_block[None, None, :, :], -1e9)

    mode = str(selector_mode).lower().strip()
    if mode not in {"topp", "top_p", "fixed_topk", "topk"}:
        raise ValueError(f"Unknown selector_mode={selector_mode!r}; use 'topp' or 'fixed_topk'.")

    # Safety bounds.
    min_topk = max(1, int(min_topk))
    if max_topk is not None:
        max_topk = max(1, min(int(max_topk), kb))
    if exact_topk is not None:
        exact_topk = max(1, min(int(exact_topk), kb))

    if mode in {"fixed_topk", "topk"}:
        topk = exact_topk if exact_topk is not None else (max_topk if max_topk is not None else int(fallback_topk))
        topk = max(1, min(int(topk), kb))
        idx = torch.topk(masked_energy, k=topk, dim=-1).indices
        mask = torch.zeros(b, h, qb, kb, dtype=torch.bool, device=device)
        mask.scatter_(-1, idx, True)
    else:
        # True top-p. Sort causal energies, compute probability mass, keep the
        # smallest prefix whose cumulative probability reaches `threshold`.
        sorted_vals, sorted_idx = torch.sort(masked_energy, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_vals, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)

        # Keep items before crossing threshold, and also include the first item
        # that crosses threshold. This implements the usual nucleus/top-p rule.
        keep_sorted = cdf <= float(threshold)
        cross = (cdf >= float(threshold)).float().argmax(dim=-1, keepdim=True)
        keep_sorted.scatter_(-1, cross, True)

        # Minimum selected prefix.
        if min_topk > 1:
            keep_sorted[..., : min(min_topk, kb)] = True
        else:
            keep_sorted[..., 0] = True

        # Optional budget cap from target_blocks_schedule. Since sorted indices
        # are in descending energy order, this keeps the best `max_topk` blocks.
        if max_topk is not None:
            cap = torch.zeros_like(keep_sorted)
            cap[..., :max_topk] = True
            keep_sorted = keep_sorted & cap
            keep_sorted[..., 0] = True

        mask = torch.zeros(b, h, qb, kb, dtype=torch.bool, device=device)
        mask.scatter_(-1, sorted_idx, keep_sorted)

    # Remove future blocks.
    mask = mask & causal_block[None, None, :, :]

    # Diagonal/current-block guard.
    offset = kb - qb
    diag_like = torch.zeros(qb, kb, dtype=torch.bool, device=device)
    for i in range(qb):
        j = i + offset
        if 0 <= j < kb:
            diag_like[i, j] = True

    mask = mask | diag_like[None, None, :, :]
    return mask.contiguous()


def hard_block_sparse_attention(q, k, v, hard_mask, block_size: int = 128):
    """
    q/k/v:
        [1, heads, seq, head_dim]

    hard_mask:
        [1, heads, q_blocks, k_blocks] bool

    纯 PyTorch block-sparse masked attention。
    """
    b, heads, q_len, head_dim = q.shape
    _, _, k_len, _ = k.shape

    if b != 1:
        raise ValueError("Only batch_size=1 is supported.")

    qb = (q_len + block_size - 1) // block_size
    kb = (k_len + block_size - 1) // block_size
    scale = head_dim ** -0.5

    # Pad to block boundary
    q_pad = qb * block_size - q_len
    k_pad = kb * block_size - k_len

    if q_pad:
        q = F.pad(q, (0, 0, 0, q_pad))

    if k_pad:
        k = F.pad(k, (0, 0, 0, k_pad))
        v = F.pad(v, (0, 0, 0, k_pad))

    q_padded = q_len + q_pad
    k_padded = k_len + k_pad

    # Full scores on padded sequence
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Block-level mask -> token-level mask
    token_mask = (
        hard_mask
        .repeat_interleave(block_size, dim=2)
        .repeat_interleave(block_size, dim=3)
    )
    token_mask = token_mask[:, :, :q_padded, :k_padded]

    scores = scores.masked_fill(~token_mask, float("-inf"))

    # Causal mask
    causal_mask = build_causal_mask(q_padded, q.device)
    causal_mask = causal_mask[:, :k_padded]

    scores = scores.masked_fill(
        causal_mask[None, None, :, :],
        float("-inf"),
    )

    # Trim to real lengths
    scores = scores[:, :, :q_len, :k_len]

    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v[:, :, :k_len, :])

    return out


def soft_surrogate_attention(
    q,
    k,
    v,
    dense_scores,
    energy,
    block_size: int = 128,
    temperature: float = 1.0,
):
    """
    可导 soft routing path。

    hard mask 本身不可导，所以 backward 用 soft surrogate。
    """
    _, _, q_len, _ = q.shape
    k_len = k.shape[2]

    energy = energy - energy.mean(dim=-1, keepdim=True)
    soft_block_mask = torch.sigmoid(energy / max(temperature, 1e-6))

    soft_token_mask = (
        soft_block_mask
        .repeat_interleave(block_size, dim=2)
        .repeat_interleave(block_size, dim=3)
    )
    soft_token_mask = soft_token_mask[:, :, :q_len, :k_len]

    scores_soft = dense_scores + torch.log(soft_token_mask + 1e-6)

    causal_mask = build_causal_mask(q_len, q.device)
    scores_soft = scores_soft.masked_fill(
        causal_mask[None, None, :, :],
        float("-inf"),
    )

    probs_soft = F.softmax(scores_soft, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs_soft, v)


def compute_mask_stats(block_scores, hard_mask):
    """
    统计 hard mask 的质量。

    block_scores:
        [1, heads, q_blocks, k_blocks]

    hard_mask:
        [1, heads, q_blocks, k_blocks]

    返回：
        density:
            在完整 q_blocks × k_blocks 矩阵上的保留比例。
            这个值会被 causal 上三角区域稀释。

        causal_density:
            只在 causal 可见区域内计算的保留比例。
            这个更接近真实有效稀疏率。

        mass_recall:
            hard_mask 保留下来的 dense attention 概率质量占比。
            越高越好，长上下文/RULER 更应该关注这个指标。
    """
    b, h, qb, kb = block_scores.shape
    device = block_scores.device

    causal_block = make_causal_block_mask(qb, kb, device)

    density = hard_mask.float().mean()

    causal_denominator = causal_block.float().sum() * b * h + 1e-8
    causal_density = hard_mask.float().sum() / causal_denominator

    mass_recall = (
        (block_scores.float() * hard_mask.float()).sum(dim=-1)
        / (block_scores.float().sum(dim=-1) + 1e-8)
    ).mean()

    return density, causal_density, mass_recall


def conv_sparse_ste_forward(
    q,
    k,
    v,
    conv_weight,
    threshold: float = 0.8,
    block_size: int = 128,
    temperature: float = 1.0,
    fallback_topk: int = 8,
    selector_mode: str = "topp",
    min_topk: int = 1,
    max_topk: int | None = None,
    exact_topk: int | None = None,
    score_stride: int = 16,
    score_chunk_size: int = 16384,
    score_sample_kernel_size: int = 7,
):
    """
    conv_weight:
        当前 layer 的所有 head kernel，shape [heads, 7, 7]

    forward:
        hard_out

    backward:
        soft_out surrogate
    """
    q_heads = q.shape[1]
    k = repeat_kv_to_q_heads(k, q_heads)
    v = repeat_kv_to_q_heads(v, q_heads)

    with torch.no_grad():
        dense_out, dense_probs, dense_scores = dense_attention_teacher(q, k, v)
        del dense_probs
        block_scores = kernels_conv_block_scores(
            q,
            k,
            block_size=block_size,
            stride=score_stride,
            chunk_size=score_chunk_size,
            sample_kernel_size=score_sample_kernel_size,
            causal=True,
        )

    energy = apply_conv_energy(block_scores.detach(), conv_weight)

    with torch.no_grad():
        hard_mask = make_hard_block_mask(
            energy.detach(),
            threshold=threshold,
            block_size=block_size,
            fallback_topk=fallback_topk,
            selector_mode=selector_mode,
            min_topk=min_topk,
            max_topk=max_topk,
            exact_topk=exact_topk,
        )

        density, causal_density, mass_recall = compute_mask_stats(
            block_scores=block_scores,
            hard_mask=hard_mask,
        )

        hard_out = hard_block_sparse_attention(
            q,
            k,
            v,
            hard_mask,
            block_size=block_size,
        )

    soft_out = soft_surrogate_attention(
        q,
        k,
        v,
        dense_scores.detach(),
        energy,
        block_size=block_size,
        temperature=temperature,
    )

    # Straight-through estimator
    pred = hard_out + (soft_out - soft_out.detach())

    # Token-level MSE loss
    mse_loss = F.mse_loss(pred.float(), dense_out.float())

    # Block-level regression loss
    # 直接监督 energy 去拟合 block_scores 的 pattern。
    b, h, qb, kb = block_scores.shape
    device = block_scores.device

    causal_block = make_causal_block_mask(qb, kb, device)
    block_target = block_scores.float() * causal_block[None, None, :, :]

    # 用 causal 区域的平均值做 scale，避免未来区域 0 过多导致 scale 偏小
    causal_scale = (
        block_target.sum()
        / (causal_block.float().sum() * b * h + 1e-8)
    )
    scale = causal_scale + 1e-8

    block_loss = F.mse_loss(
        energy.float() / scale,
        block_target.detach() / scale,
    )

    loss = mse_loss + 0.1 * block_loss

    stats = {
        "density": density.detach(),
        "causal_density": causal_density.detach(),
        "mass_recall": mass_recall.detach(),
        "energy_mean": energy.mean().detach(),
        "energy_std": energy.std().detach(),
        "mse_loss": mse_loss.detach(),
        "block_loss": block_loss.detach(),
    }

    return pred, dense_out, loss, stats
