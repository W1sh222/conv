import torch
import math
import torch.nn.functional as F
from xattn.src.utils import *
from xattn.src.kernels import (
    flat_group_gemm_fuse_reshape,
    softmax_fuse_block_sum,
)
from block_sparse_attn import block_sparse_attn_func
import os


# Global cache for conv weights to avoid repeated disk I/O
_CONV_WEIGHT_CACHE = {}


# Default paths for conv kernel weights
_CONV_WEIGHT_DIR = os.path.join(os.path.dirname(__file__), "..", "conv_weights")
_DEFAULT_WEIGHT_PATH = os.path.join(_CONV_WEIGHT_DIR, "conv_kernel_7x7_2w_1e5.pt")


def _get_conv_weight(kernel_size=7, weight_path=None, device=None):
    """
    支持这些权重格式：

    1. [1, 1, K, K]
       全局共享 kernel

    2. [H, K, K]
       每个 head 一个 kernel，所有 layer 共享

    3. [H, 1, K, K]
       每个 head 一个 kernel，所有 layer 共享，带 conv channel 维

    4. [L, H, K, K]
       每个 layer、每个 head 一个 kernel。你现在的 [32,32,7,7] 就是这个格式。

    5. [L, H, 1, K, K]
       每个 layer、每个 head 一个 kernel，带 conv channel 维
    """
    cache_key = (weight_path or _DEFAULT_WEIGHT_PATH, str(device))
    if cache_key in _CONV_WEIGHT_CACHE:
        return _CONV_WEIGHT_CACHE[cache_key]

    weight = None
    if weight_path is not None and os.path.exists(weight_path):
        weight = torch.load(weight_path, map_location=device, weights_only=True)
    elif weight_path is None and os.path.exists(_DEFAULT_WEIGHT_PATH):
        weight = torch.load(_DEFAULT_WEIGHT_PATH, map_location=device, weights_only=True)

    if weight is None:
        weight = torch.ones(
            1,
            1,
            kernel_size,
            kernel_size,
            device=device,
            dtype=torch.float32,
        ) / (kernel_size * kernel_size)

    if not torch.is_tensor(weight):
        raise TypeError(f"conv weight must be torch.Tensor, got {type(weight)}")

    shape = tuple(weight.shape)

    if weight.dim() == 2:
        # [K,K]
        if shape != (kernel_size, kernel_size):
            raise ValueError(f"Bad conv weight shape {shape}")

    elif weight.dim() == 3:
        # [H,K,K]
        if shape[-2:] != (kernel_size, kernel_size):
            raise ValueError(f"Bad conv weight shape {shape}")

    elif weight.dim() == 4:
        # [1,1,K,K], [H,1,K,K], or [L,H,K,K]
        if shape[-2:] != (kernel_size, kernel_size):
            raise ValueError(f"Bad conv weight shape {shape}")

    elif weight.dim() == 5:
        # [L,H,1,K,K]
        if shape[-3:] != (1, kernel_size, kernel_size):
            raise ValueError(f"Bad conv weight shape {shape}")

    else:
        raise ValueError(
            f"Unsupported conv weight shape {shape}. "
            f"Expected [K,K], [H,K,K], [1,1,K,K], [H,1,K,K], [L,H,K,K], or [L,H,1,K,K]."
        )

    if device is not None:
        weight = weight.to(device)

    _CONV_WEIGHT_CACHE[cache_key] = weight
    return weight


def _threshold_to_float(threshold, default=0.8):
    if threshold is None:
        return float(default)
    if isinstance(threshold, torch.Tensor):
        return float(threshold.detach().float().mean().cpu().item())
    return float(threshold)


def _safe_causal_topk_mask(
    energy,
    threshold=0.8,
    fallback_topk=8,
    causal=True,
    keep_sink=False,
    keep_recent=False,
):
    """
    安全 fallback mask 构造。

    energy: [batch, heads, q_blocks, k_blocks]

    不调用 find_blocks_chunked，避免某些 shape / CUDA kernel 触发 illegal memory。
    每个 query block 选择 top-k 个 key block，并强制 causal。
    """
    b, h, qb, kb = energy.shape
    device = energy.device

    energy = torch.nan_to_num(
        energy.float(),
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    )

    threshold_f = _threshold_to_float(threshold, default=0.8)

    # threshold 越高，保留越少；同时至少保留 fallback_topk 个 block
    dynamic_k = int(math.ceil(kb * max(1e-3, 1.0 - threshold_f)))
    topk = max(int(fallback_topk), dynamic_k)
    topk = max(1, min(topk, kb))

    if causal:
        # 允许 j <= i + offset
        offset = kb - qb
        causal_block = torch.tril(
            torch.ones(qb, kb, dtype=torch.bool, device=device),
            diagonal=offset,
        )
        masked_energy = energy.masked_fill(
            ~causal_block[None, None, :, :],
            -1e9,
        )
    else:
        causal_block = torch.ones(qb, kb, dtype=torch.bool, device=device)
        masked_energy = energy

    idx = torch.topk(masked_energy, k=topk, dim=-1).indices

    mask = torch.zeros(b, h, qb, kb, dtype=torch.bool, device=device)
    mask.scatter_(-1, idx, True)

    if causal:
        mask = mask & causal_block[None, None, :, :]

    # 保底：每个 q block 至少保留对应的对角 block
    offset = kb - qb
    diag_like = torch.zeros(qb, kb, dtype=torch.bool, device=device)
    for i in range(qb):
        j = i + offset
        if 0 <= j < kb:
            diag_like[i, j] = True
    mask = mask | diag_like[None, None, :, :]

    if causal:
        mask = mask & causal_block[None, None, :, :]

    if keep_sink:
        mask[:, :, :, 0] = True

    if keep_recent:
        recent = torch.zeros(qb, kb, dtype=torch.bool, device=device)
        for i in range(qb):
            j = i + offset
            if 0 <= j < kb:
                recent[i, j] = True
        mask = mask | recent[None, None, :, :]

    return mask.contiguous()


def _select_conv_weight_for_layer_head(
    weight,
    num_heads,
    layer_idx=None,
    batch_size=1,
    kernel_size=7,
):
    """
    返回 grouped conv2d 需要的 weight：

        [batch_size * num_heads, 1, K, K]

    支持输入：
        [K, K]
        [H, K, K]
        [1, 1, K, K]
        [H, 1, K, K]
        [L, H, K, K]
        [L, H, 1, K, K]
    """
    shape = tuple(weight.shape)

    # [K,K]：所有 layer/head 共享
    if weight.dim() == 2:
        selected = weight[None, None, :, :].expand(
            num_heads, 1, kernel_size, kernel_size
        )

    # [H,K,K]：每个 head 不同，所有 layer 共享
    elif weight.dim() == 3:
        if weight.shape[0] == 1:
            selected = weight.expand(num_heads, kernel_size, kernel_size)[:, None, :, :]
        elif weight.shape[0] == num_heads:
            selected = weight[:, None, :, :]
        else:
            raise ValueError(
                f"Weight head dim={weight.shape[0]} does not match num_heads={num_heads}. "
                f"weight shape={shape}"
            )

    elif weight.dim() == 4:
        # [1,1,K,K]：所有 layer/head 共享
        if weight.shape[0] == 1 and weight.shape[1] == 1:
            selected = weight.expand(num_heads, 1, kernel_size, kernel_size)

        # [H,1,K,K]：每个 head 不同，所有 layer 共享
        elif weight.shape[0] == num_heads and weight.shape[1] == 1:
            selected = weight

        # [L,H,K,K]：每层每头不同，也就是 [32,32,7,7]
        else:
            if layer_idx is None:
                raise ValueError(
                    f"layer_idx is required for per-layer weight shape {shape}"
                )

            num_layers = weight.shape[0]
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(
                    f"layer_idx={layer_idx} out of range for num_layers={num_layers}"
                )

            if weight.shape[1] == num_heads:
                selected = weight[layer_idx][:, None, :, :]
            elif weight.shape[1] == 1:
                selected = weight[layer_idx].expand(
                    num_heads, kernel_size, kernel_size
                )[:, None, :, :]
            else:
                raise ValueError(
                    f"Weight head dim={weight.shape[1]} does not match num_heads={num_heads}. "
                    f"weight shape={shape}"
                )

    # [L,H,1,K,K]
    elif weight.dim() == 5:
        if layer_idx is None:
            raise ValueError(
                f"layer_idx is required for per-layer weight shape {shape}"
            )

        num_layers = weight.shape[0]
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(
                f"layer_idx={layer_idx} out of range for num_layers={num_layers}"
            )

        if weight.shape[1] == num_heads:
            selected = weight[layer_idx]
        elif weight.shape[1] == 1:
            selected = weight[layer_idx].expand(
                num_heads, 1, kernel_size, kernel_size
            )
        else:
            raise ValueError(
                f"Weight head dim={weight.shape[1]} does not match num_heads={num_heads}. "
                f"weight shape={shape}"
            )

    else:
        raise ValueError(f"Unsupported weight dim={weight.dim()}, shape={shape}")

    selected = selected.contiguous()

    if batch_size > 1:
        selected = selected.repeat(batch_size, 1, 1, 1)

    return selected.contiguous()


def apply_conv2d_block_map(
    block_scores,
    kernel_size=7,
    weight_path=None,
    layer_idx=None,
):
    """
    Apply 2D convolution smoothing on block importance map.

    Args:
        block_scores: [batch, heads, q_blocks, k_blocks]
        weight_path:
            支持 .pt shape:
                [1,1,K,K]
                [H,1,K,K]
                [L,H,1,K,K]
        layer_idx:
            当权重是 [L,H,1,K,K] 时必须传入

    Returns:
        smoothed block_scores: [batch, heads, q_blocks, k_blocks]
    """
    b, h, qb, kb = block_scores.shape
    device = block_scores.device

    x = block_scores
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        x = x.float()

    # grouped conv 输入：[1, b*h, qb, kb]
    x = x.reshape(1, b * h, qb, kb)

    weight = _get_conv_weight(
        kernel_size=kernel_size,
        weight_path=weight_path,
        device=device,
    ).to(dtype=x.dtype)

    weight = _select_conv_weight_for_layer_head(
        weight=weight,
        num_heads=h,
        layer_idx=layer_idx,
        batch_size=b,
        kernel_size=kernel_size,
    ).to(device=device, dtype=x.dtype)

    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")

    # groups=b*h：每个 batch/head 用自己的 7x7 kernel
    smoothed = F.conv2d(
        x_padded,
        weight,
        stride=1,
        padding=0,
        groups=b * h,
    )

    return smoothed.view(b, h, qb, kb).to(block_scores.dtype)


def conv_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    conv_kernel_size: int = 7,
    conv_weight_path=None,
    layer_idx=None,
    conv_safe_topk=False,
    fallback_topk=8,
):
    """
    Block importance estimation using direct block-sum + 2D convolution smoothing.

    Differences from xattn_estimate:
    1. Computes block importance by directly summing attention weights per block
       (no antidiagonal scoring trick).
    2. Applies 2D convolution (kernel=conv_kernel_size, replicate padding) on
       the full [q_blocks, k_blocks] importance map to smooth scores.
    3. Uses the smoothed scores for block selection via find_blocks_chunked().

    Args:
        conv_weight_path: optional path to .pt file with conv weight (1x1xKxK)

    Returns:
        attn_sums: shape [batch, heads, q_block_num, k_block_num]
        simple_masks: shape [batch, heads, q_block_num, k_block_num]
    """
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to("cuda")
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(
            "cuda"
        )
    else:
        pad_query_states = query_states

    assert num_kv_head == num_q_head

    if use_triton and (
        "100" not in torch.cuda.get_device_properties(torch.cuda.current_device()).name
    ):
        use_triton = False
        print(
            "setting use triton to false. Triton kernel not surpported on this device"
        )

    # ── Step 1: reshape Q/K with stride (non-triton only; same as xattn) ──
    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    if not use_triton:
        if select_mode == "inverse" or select_mode == "":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: (stride * kdb), :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
            )
        elif select_mode == "double":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [
                    reshaped_key[:, :, :, -head_dim:],
                    reshaped_key[:, :, :, 0:-head_dim],
                ],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    # ── Step 2: compute block importance per chunk (direct block sum) ──
    attn_sum_list = []

    for chunk_idx in range(q_chunk_num):
        if use_triton:
            if kdb != 1:
                raise ValueError("use_triton and kdb cannot be used together")
            attn_weights_slice = flat_group_gemm_fuse_reshape(
                pad_query_states[
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size)
                    * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                    * stride,
                    :,
                ],
                pad_key_states,
                stride,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                is_causal=causal,
            )
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                is_causal=causal,
            )
        else:
            chunked_query = reshaped_query[
                :,
                :,
                (chunk_idx * reshaped_chunk_size)
                // kdb : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                // kdb,
                :,
            ]
            attn_weights_slice = torch.matmul(
                chunked_query,
                reshaped_key.transpose(2, 3),
            ).to("cuda")

            attn_weights_slice = (
                attn_weights_slice / math.sqrt(head_dim) / stride / norm
            )

            if causal:
                causal_mask = torch.zeros(
                    (
                        batch_size,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size * k_chunk_num,
                    ),
                    device=key_states.device,
                )
                causal_mask[:, :, :, (-k_reshaped_num_to_pad) :] = float("-inf")
                chunk_start = (
                    chunk_idx + offset_token_chunk_num
                ) * reshaped_chunk_size
                chunk_end = chunk_start + reshaped_chunk_size
                causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                    torch.ones(
                        1,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size,
                        device=key_states.device,
                    )
                    * float("-inf"),
                    diagonal=1,
                )

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    causal_mask[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = float(
                        "-inf"
                    )

                causal_mask[:, :, :, chunk_end:] = float("-inf")
                causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
                attn_weights_slice = attn_weights_slice + causal_mask.to(
                    attn_weights_slice.device
                )

            if softmax:
                attn_weights_slice = F.softmax(
                    attn_weights_slice, dim=-1, dtype=torch.float32
                ).to(pad_query_states.dtype)
            else:
                attn_weights_slice = torch.exp(attn_weights_slice).to(
                    pad_query_states.dtype
                )
            attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = 0

            # Direct block sum: sum attention weights within each block
            attn_sum = (
                attn_weights_slice.view(
                    batch_size,
                    num_kv_head,
                    num_blocks_per_chunk,
                    reshaped_block_size // kdb,
                    -1,
                    reshaped_block_size,
                )
                .sum(dim=-1)
                .sum(dim=-2)
                .to("cuda")
            )
            del chunked_query

        attn_sum_list.append(attn_sum)
        del attn_weights_slice

    if not use_triton:
        del reshaped_query, reshaped_key

    # ── Step 3: concatenate and apply 2D convolution smoothing ──
    attn_sums = torch.cat(attn_sum_list, dim=-2)  # [b, h, q_block_num, k_block_num]
    attn_sums_smoothed = apply_conv2d_block_map(
        attn_sums,
        kernel_size=conv_kernel_size,
        weight_path=conv_weight_path,
        layer_idx=layer_idx,
    )

    # ── Step 4: block selection (one call on full map, same as per-chunk concat) ──
    attn_sums_smoothed = torch.nan_to_num(
        attn_sums_smoothed,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    if conv_safe_topk:
        simple_masks = _safe_causal_topk_mask(
            attn_sums_smoothed,
            threshold=threshold,
            fallback_topk=fallback_topk,
            causal=causal,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
        )
    else:
        try:
            simple_masks = find_blocks_chunked(
                attn_sums_smoothed,
                k_block_num - q_block_num,
                threshold,
                None,
                decoding=False,
                mode="prefill",
                causal=causal,
            )

            # 尽早暴露异步 CUDA 错误，避免错误延迟到后面随机位置
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        except Exception as e:
            print(
                f"[Conv.py WARN] find_blocks_chunked failed, "
                f"fallback to safe causal top-k mask. error={repr(e)}"
            )

            simple_masks = _safe_causal_topk_mask(
                attn_sums_smoothed,
                threshold=threshold,
                fallback_topk=fallback_topk,
                causal=causal,
                keep_sink=keep_sink,
                keep_recent=keep_recent,
            )

    # ── Step 5: post-processing (same as xattn) ──
    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num, q_block_num, dtype=bool, device=simple_masks.device
                ),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )
    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_kv_head, q_block_num, q_block_num)
        )
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )

    return attn_sums, simple_masks


def Conv_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
    conv_kernel_size=7,
    conv_weight_path=None,
    layer_idx=None,
    conv_safe_topk=False,
    fallback_topk=8,
):
    """
    Conv-based sparse prefill attention.

    Same interface as Xattention_prefill, but replaces the antidiagonal scoring
    with direct block-sum + 2D convolution smoothing for block selection.
    """
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            )
        )

    attn_sums, approx_simple_mask = conv_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=use_triton,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        conv_kernel_size=conv_kernel_size,
        conv_weight_path=conv_weight_path,
        layer_idx=layer_idx,
        conv_safe_topk=conv_safe_topk,
        fallback_topk=fallback_topk,
    )

    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    # ── Reuse block_sparse_attn_func (same as Xattention_prefill) ──
    assert block_size == 128
    assert batch_size == 1
    query_states = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
    key_states = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    value_states = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    q_cu_seq_lens = torch.tensor(
        [0, q_len], dtype=torch.int32, device=query_states.device
    )
    k_cu_seq_lens = torch.tensor(
        [0, k_len], dtype=torch.int32, device=query_states.device
    )
    head_mask_type = torch.tensor(
        [1 for _ in range(num_heads)], device=query_states.device, dtype=torch.int32
    )
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device

    attn_output = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )
    attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(
        1, 2
    )

    del query_states, approx_simple_mask, attn_sums
    return attn_output