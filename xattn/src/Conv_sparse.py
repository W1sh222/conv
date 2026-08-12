import torch
import math
import torch.nn.functional as F
from xattn.src.utils import *
from xattn.src.kernels_conv import (
    flat_group_gemm_fuse_reshape,
    softmax_fuse_block_sum,
)
from block_sparse_attn import block_sparse_attn_func
import os


# Global caches to avoid repeated disk I/O / small tensor allocations.
_CONV_WEIGHT_CACHE = {}
_STATIC_PREFILL_TENSOR_CACHE = {}


# Default paths for conv kernel weights
_CONV_WEIGHT_DIR = os.path.join(os.path.dirname(__file__), "..", "conv_weights")
_DEFAULT_WEIGHT_PATH = "/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn/conv_weights/.pt"


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
    if weight_path is None:
        weight_path = _DEFAULT_WEIGHT_PATH

    cache_key = (str(weight_path), str(device))
    if cache_key in _CONV_WEIGHT_CACHE:
        return _CONV_WEIGHT_CACHE[cache_key]

    weight = None
    
    weight_path = "/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn/conv_weights/conv_kernel_7x7_ruler_mix_guarded_t06_top16_1e4_step9500.pt"

    if weight_path is not None and os.path.exists(weight_path):
        print("loading weight_path:", weight_path)
        weight = torch.load(weight_path, map_location=device, weights_only=True)
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



def _fixed_topk_mask_from_scores(
    scores: torch.Tensor,
    fixed_topk: int,
    offset: int,
    causal: bool = True,
):
    """
    Select a fixed per-query top-k budget from block scores.

    Args:
        scores: [batch, heads, q_blocks_this_chunk, k_blocks]
        fixed_topk: requested number of key blocks per query block
        offset: causal block offset, identical to find_blocks_chunked's offset
        causal: whether to enforce block-causal validity

    Notes:
        For early causal query blocks with fewer than k valid key blocks,
        all valid blocks are kept. A diagonal-like fallback guarantees that
        every query block has at least one valid key block.
    """
    b, h, qb, kb = scores.shape
    device = scores.device

    topk = int(fixed_topk)
    if topk <= 0:
        raise ValueError(f"fixed_topk must be positive, got {fixed_topk}")
    topk = min(topk, kb)

    energy = torch.nan_to_num(
        scores.float(),
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    )

    if causal:
        q_idx = torch.arange(qb, device=device)[:, None]
        k_idx = torch.arange(kb, device=device)[None, :]
        causal_block = k_idx <= (q_idx + int(offset))
        energy = energy.masked_fill(
            ~causal_block[None, None, :, :],
            -1e9,
        )
    else:
        causal_block = torch.ones(qb, kb, dtype=torch.bool, device=device)

    idx = torch.topk(energy, k=topk, dim=-1).indices
    mask = torch.zeros(b, h, qb, kb, dtype=torch.bool, device=device)
    mask.scatter_(-1, idx, True)

    if causal:
        mask = mask & causal_block[None, None, :, :]

    # Guarantee at least one valid block per query block.
    fallback_k = (torch.arange(qb, device=device) + int(offset)).clamp(0, kb - 1)
    fallback = torch.zeros(qb, kb, dtype=torch.bool, device=device)
    fallback[torch.arange(qb, device=device), fallback_k] = True
    if causal:
        fallback = fallback & causal_block

    empty = mask.sum(dim=-1) == 0
    mask = mask | (empty[:, :, :, None] & fallback[None, None, :, :])

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



def _choose_softmax_segment_size(k_reshaped_seq_len: int, reshaped_block_size: int) -> int:
    """
    Choose a larger segment for softmax_fuse_block_sum without changing the
    mathematical result.

    Requirements from the Triton kernel:
        1. segment_size % reshaped_block_size == 0
        2. k_reshaped_seq_len % segment_size == 0

    The old code effectively used segment_size=reshaped_block_size, which creates
    many tiny segments. 64/128 usually reduce loop overhead while keeping register
    pressure reasonable.
    """
    target = int(os.environ.get("CONV_SOFTMAX_SEGMENT_SIZE", "128"))
    target = max(int(reshaped_block_size), min(target, int(k_reshaped_seq_len)))

    seg = (target // int(reshaped_block_size)) * int(reshaped_block_size)
    while seg >= int(reshaped_block_size):
        if int(k_reshaped_seq_len) % seg == 0:
            return int(seg)
        seg -= int(reshaped_block_size)

    return int(reshaped_block_size)


def _get_static_prefill_tensors(q_len: int, k_len: int, num_heads: int, device):
    """
    Cache tiny static tensors used by block_sparse_attn_func.

    This avoids repeated torch.tensor allocations for every layer/prefill call.
    """
    key = (int(q_len), int(k_len), int(num_heads), str(device))
    cached = _STATIC_PREFILL_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached

    q_cu_seq_lens = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    k_cu_seq_lens = torch.tensor([0, k_len], dtype=torch.int32, device=device)
    head_mask_type = torch.ones(num_heads, device=device, dtype=torch.int32)

    cached = (q_cu_seq_lens, k_cu_seq_lens, head_mask_type)
    _STATIC_PREFILL_TENSOR_CACHE[key] = cached
    return cached



def _compute_average_density_from_mask(
    mask: torch.Tensor,
    q_block_num: int,
    k_block_num: int,
    causal: bool = True,
):
    """
    Compute average block density from the actual sparse mask.

    Density definition:
        selected valid blocks / all theoretically computable blocks.

    For causal prefill, the denominator is the number of lower-triangular
    block positions, generalized to q_blocks != k_blocks by the offset
    k_block_num - q_block_num. For non-causal prefill, the denominator is
    q_block_num * k_block_num per batch/head.

    This intentionally calls .item(), so it synchronizes CUDA only when the
    caller explicitly sets return_density=True.
    """
    mask = mask[:, :, :q_block_num, :k_block_num]
    batch_size, num_heads, _, _ = mask.shape

    selected = mask.sum(dtype=torch.float32)

    if causal:
        offset = k_block_num - q_block_num
        valid_per_q = torch.arange(
            q_block_num,
            device=mask.device,
            dtype=torch.float32,
        ) + float(offset + 1)
        valid_per_q = torch.clamp(valid_per_q, min=0.0, max=float(k_block_num))
        valid_per_head = valid_per_q.sum()
        total = valid_per_head * float(batch_size * num_heads)
    else:
        total = torch.tensor(
            float(batch_size * num_heads * q_block_num * k_block_num),
            device=mask.device,
            dtype=torch.float32,
        )

    density = selected / torch.clamp(total, min=1.0)
    return float(density.detach().float().cpu().item())


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
    fixed_topk=None,
    topk=None,
):
    """
    Block importance estimation using direct block-sum + 2D convolution smoothing.

    Differences from xattn_estimate:
    1. Computes block importance by directly summing attention weights per block
       (no antidiagonal scoring trick).
    2. Applies 2D convolution (kernel=conv_kernel_size, replicate padding) on
       the full [q_blocks, k_blocks] importance map to smooth scores.
    3. Uses the smoothed scores for block selection via the same chunk-wise
       find_blocks_chunked schedule as xattn_estimate, so sparsity/kept-block
       counts are as aligned as possible under the same threshold.

    Args:
        conv_weight_path: optional path to .pt file with conv weight (1x1xKxK)

    Returns:
        attn_sums: shape [batch, heads, q_block_num, k_block_num]
        simple_masks: shape [batch, heads, q_block_num, k_block_num]
    """
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    # Accept both `fixed_topk=` (benchmark compatibility) and `topk=` (short alias).
    if topk is not None:
        if fixed_topk is not None and int(fixed_topk) != int(topk):
            raise ValueError(
                f"Conflicting top-k values: fixed_topk={fixed_topk}, topk={topk}"
            )
        fixed_topk = int(topk)

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

    # The conv-spaced kernel keeps roughly 1 / conv_kernel_size of the
    # original inverse anti-diagonal offsets while preserving the original
    # 128x128 tile shape. Renormalize logits by the number of sampled offsets
    # instead of the full stride; otherwise the logits become too small.
    diag_sample_kernel_size = max(1, min(int(conv_kernel_size), int(stride)))
    sampled_stride_norm = max(1.0, float(stride) / float(diag_sample_kernel_size))
    softmax_segment_size = _choose_softmax_segment_size(
        k_reshaped_seq_len=k_reshaped_seq_len,
        reshaped_block_size=reshaped_block_size,
    )

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
                sample_kernel_size=diag_sample_kernel_size,
            )
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                softmax_segment_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / sampled_stride_norm / norm,
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
    if len(attn_sum_list) == 1:
        attn_sums = attn_sum_list[0]
    else:
        attn_sums = torch.cat(attn_sum_list, dim=-2)  # [b, h, q_block_num, k_block_num]

    attn_sums_smoothed = apply_conv2d_block_map(
        attn_sums,
        kernel_size=conv_kernel_size,
        weight_path=conv_weight_path,
        layer_idx=layer_idx,
    )

    # ── Step 4: block selection ──
    # Fair-comparison path:
    #   xattn_estimate calls find_blocks_chunked once per query chunk and passes
    #   offset = k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk.
    #   Do the same here, but on the convolution-smoothed block scores. This keeps
    #   threshold semantics, chunk offsets, and kept-block counts as close as
    #   possible to xattn while still selecting blocks according to conv scores.
    attn_sums_smoothed = torch.nan_to_num(
        attn_sums_smoothed,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    if fixed_topk is not None:
        # Fair fixed-budget path: same per-query top-k selection rule as XAttention.
        simple_mask_list = []
        for chunk_idx in range(q_chunk_num):
            chunk_start = chunk_idx * num_blocks_per_chunk
            chunk_end = (chunk_idx + 1) * num_blocks_per_chunk
            mask_offset = (
                k_block_num
                - q_block_num
                + chunk_idx * num_blocks_per_chunk
            )

            simple_mask = _fixed_topk_mask_from_scores(
                attn_sums_smoothed[:, :, chunk_start:chunk_end, :],
                fixed_topk=int(fixed_topk),
                offset=mask_offset,
                causal=causal,
            )
            simple_mask_list.append(simple_mask)

        simple_masks = (
            simple_mask_list[0]
            if len(simple_mask_list) == 1
            else torch.cat(simple_mask_list, dim=-2)
        )

    elif conv_safe_topk:
        # Explicit debug/safety mode only.
        simple_masks = _safe_causal_topk_mask(
            attn_sums_smoothed,
            threshold=threshold,
            fallback_topk=fallback_topk,
            causal=causal,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
        )
    else:
        simple_mask_list = []
        try:
            for chunk_idx in range(q_chunk_num):
                chunk_start = chunk_idx * num_blocks_per_chunk
                chunk_end = (chunk_idx + 1) * num_blocks_per_chunk

                simple_mask = find_blocks_chunked(
                    attn_sums_smoothed[:, :, chunk_start:chunk_end, :],
                    k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
                    threshold,
                    None,
                    decoding=False,
                    mode="prefill",
                    causal=causal,
                )
                simple_mask_list.append(simple_mask)

            simple_masks = (
                simple_mask_list[0]
                if len(simple_mask_list) == 1
                else torch.cat(simple_mask_list, dim=-2)
            )

            # 尽早暴露异步 CUDA 错误，避免错误延迟到后面随机位置
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()

        except Exception as e:
            # Do not silently fall back during fair comparison; safe top-k changes
            # sparsity and kept-block counts. If needed, rerun with --conv_safe_topk.
            print(
                f"[Conv.py WARN] chunk-wise find_blocks_chunked failed. "
                f"To use the old safe top-k fallback, rerun with --conv_safe_topk. "
                f"error={repr(e)}"
            )
            raise

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
        simple_masks[:, :, :, 0] = True
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


def _sanitize_block_sparse_mask(
    mask: torch.Tensor,
    q_block_num: int,
    k_block_num: int,
    causal: bool = True,
    keep_sink: bool = False,
    keep_recent: bool = False,
):
    """
    block_sparse_attn_func 前的安全清洗。

    This version keeps the same mask semantics as the original version, but avoids
    CPU-GPU synchronization from Tensor.any() inside Python control flow.
    """
    mask = mask[:, :, :q_block_num, :k_block_num].to(torch.bool).contiguous()
    device = mask.device

    b, h, qb, kb = mask.shape
    assert qb == q_block_num, (qb, q_block_num)
    assert kb == k_block_num, (kb, k_block_num)

    offset = k_block_num - q_block_num
    q_idx = torch.arange(q_block_num, device=device)[:, None]
    k_idx = torch.arange(k_block_num, device=device)[None, :]

    if causal:
        causal_block = k_idx <= (q_idx + offset)
        mask = mask & causal_block[None, None, :, :]

    if keep_sink and k_block_num > 0:
        # sink 是 key block 0，不是 query block 0
        mask[:, :, :, 0] = True

    # Same fallback target as the original Python loop:
    # kj = max(0, min(qi + offset, k_block_num - 1))
    fallback_k = (torch.arange(q_block_num, device=device) + offset).clamp(
        0, k_block_num - 1
    )
    fallback = torch.zeros(
        q_block_num,
        k_block_num,
        dtype=torch.bool,
        device=device,
    )
    fallback[torch.arange(q_block_num, device=device), fallback_k] = True

    if keep_recent:
        mask = mask | fallback[None, None, :, :]

    # 保底：每个 q block 至少保留一个合法 key block。
    # GPU-only; no `if empty.any()` to avoid implicit cuda synchronization.
    empty = mask.sum(dim=-1) == 0  # [b, h, q_block_num]
    mask = mask | (empty[:, :, :, None] & fallback[None, None, :, :])

    return mask.contiguous()

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
    return_density=False,
    fixed_topk=None,
    topk=None,
):
    """
    Conv-based sparse prefill attention.

    Same interface as Xattention_prefill, but replaces the antidiagonal scoring
    with direct block-sum + 2D convolution smoothing for block selection.

    If return_density=True, return:
        (attn_output, average_density)
    where average_density is selected_blocks / all valid causal blocks.
    """
    # print("conv_kernel_size:", conv_kernel_size)
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size

    # Accept `topk=` as an alias while keeping `fixed_topk=` compatible with
    # the existing same-topk benchmark script.
    if topk is not None:
        if fixed_topk is not None and int(fixed_topk) != int(topk):
            raise ValueError(
                f"Conflicting top-k values: fixed_topk={fixed_topk}, topk={topk}"
            )
        fixed_topk = int(topk)

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
        fixed_topk=fixed_topk,
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
    q_cu_seq_lens, k_cu_seq_lens, head_mask_type = _get_static_prefill_tensors(
        q_len=q_len,
        k_len=k_len,
        num_heads=num_heads,
        device=query_states.device,
    )
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device

    mask = _sanitize_block_sparse_mask(
        approx_simple_mask,
        q_block_num=q_block_num,
        k_block_num=k_block_num,
        causal=causal,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
    )
    
    # selected_blocks = mask.sum().item()

    # num_to_compute = (
    #     (k_block_num + 1)
    #     * k_block_num
    #     / 2
    #     * num_heads
    # )

    # density = selected_blocks / num_to_compute

    # print(
    #     f"[Conv] "
    #     f"selected={selected_blocks:.0f} "
    #     f"total={num_to_compute:.0f} "
    #     f"density={density*100:.2f}%"
    # )

    assert mask.device == query_states.device
    assert mask.dtype == torch.bool
    assert mask.shape == (batch_size, num_heads, q_block_num, k_block_num), mask.shape
    if os.environ.get("CONV_DEBUG_ASSERT", "0") == "1":
        assert mask.sum(dim=-1).min().item() > 0, "Some query block has empty key blocks"

    average_density = None
    if return_density:
        average_density = _compute_average_density_from_mask(
            mask,
            q_block_num=q_block_num,
            k_block_num=k_block_num,
            causal=causal,
        )

    attn_output = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        mask,
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
    if return_density:
        return attn_output, average_density
    return attn_output