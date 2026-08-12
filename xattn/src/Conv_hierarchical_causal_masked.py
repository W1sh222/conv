import torch
import math
import torch.nn.functional as F
from xattn.src.utils import *
from xattn.src.kernels_conv_hierarchical_sparse_causal_masked import (
    flat_group_gemm_fuse_reshape,
    softmax_fuse_block_sum,
    sparse_softmax_fuse_block_sum,
)
from block_sparse_attn import block_sparse_attn_func
import os


# Global caches to avoid repeated disk I/O / small tensor allocations.
_CONV_WEIGHT_CACHE = {}
_STATIC_PREFILL_TENSOR_CACHE = {}
_CONV_VALID_MASK_CACHE = {}
_CONV_SUPPORT_CACHE = {}


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
    
    weight_path = "/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn/conv_weights/conv_kernel_7x7_ruler_mix_guarded_l1_t06_top16_1e4_step16000.pt"

    weight = None

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



def _build_prefill_valid_block_mask(
    q_blocks: int,
    k_blocks: int,
    real_q_blocks: int,
    real_k_blocks: int,
    device,
    causal: bool = True,
):
    """
    Build the structural validity mask for a prefill block-score map.

    The estimator may operate on a padded [q_blocks, k_blocks] grid. This mask
    marks only real query/key blocks as valid and, for causal prefill, enforces
    the global block-causal relation

        k_idx <= q_idx + (real_k_blocks - real_q_blocks).

    Returned shape:
        [1, 1, q_blocks, k_blocks] bool
    """
    q_blocks = int(q_blocks)
    k_blocks = int(k_blocks)
    real_q_blocks = max(0, min(int(real_q_blocks), q_blocks))
    real_k_blocks = max(0, min(int(real_k_blocks), k_blocks))

    q_idx = torch.arange(q_blocks, device=device)[:, None]
    k_idx = torch.arange(k_blocks, device=device)[None, :]

    valid = (
        (q_idx < real_q_blocks)
        & (k_idx < real_k_blocks)
    )

    if causal:
        offset = real_k_blocks - real_q_blocks
        valid = valid & (k_idx <= (q_idx + int(offset)))

    return valid[None, None, :, :].contiguous()


def _get_prefill_valid_block_mask_cached(
    q_blocks: int,
    k_blocks: int,
    real_q_blocks: int,
    real_k_blocks: int,
    device,
    causal: bool = True,
):
    key = (
        int(q_blocks),
        int(k_blocks),
        int(real_q_blocks),
        int(real_k_blocks),
        str(device),
        bool(causal),
    )
    cached = _CONV_VALID_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    valid = _build_prefill_valid_block_mask(
        q_blocks=q_blocks,
        k_blocks=k_blocks,
        real_q_blocks=real_q_blocks,
        real_k_blocks=real_k_blocks,
        device=device,
        causal=causal,
    )
    _CONV_VALID_MASK_CACHE[key] = valid
    return valid


def _get_conv_support_cached(
    valid_ch: torch.Tensor,
    weight: torch.Tensor,
    kernel_size: int,
    batch_size: int,
    num_heads: int,
    q_blocks: int,
    k_blocks: int,
    real_q_blocks: int,
    real_k_blocks: int,
    causal: bool,
    layer_idx,
    weight_path,
    raw_weight_cache_id,
    eps: float,
):
    """
    Cache Conv(V, W), the masked-normalization denominator.

    It depends only on geometry, causal support and selected conv weights, not on
    the current block score tensor. Reusing it removes one grouped conv2d from
    every repeated prefill/layer call with the same configuration.
    """
    key = (
        str(valid_ch.device),
        int(batch_size),
        int(num_heads),
        int(q_blocks),
        int(k_blocks),
        int(real_q_blocks),
        int(real_k_blocks),
        bool(causal),
        int(kernel_size),
        None if layer_idx is None else int(layer_idx),
        str(weight_path),
        tuple(weight.shape),
        str(weight.dtype),
        int(raw_weight_cache_id),
    )
    cached = _CONV_SUPPORT_CACHE.get(key)
    if cached is not None:
        return cached

    pad = int(kernel_size) // 2
    valid_padded = F.pad(
        valid_ch,
        (pad, pad, pad, pad),
        mode="constant",
        value=0.0,
    )
    support = F.conv2d(
        valid_padded,
        weight,
        stride=1,
        padding=0,
        groups=int(batch_size) * int(num_heads),
    ).clamp_min(float(eps))

    _CONV_SUPPORT_CACHE[key] = support
    return support


def apply_conv2d_block_map(
    block_scores,
    kernel_size=7,
    weight_path=None,
    layer_idx=None,
    causal=True,
    real_q_block_num=None,
    real_k_block_num=None,
    eps=1e-6,
):
    """
    Causal-aware masked normalized convolution on a prefill block-score map.

    This replaces replicate padding with structural masking and local
    renormalization.

    Let V be the valid prefill support:
        - real (non-padding) query blocks only
        - real (non-padding) key blocks only
        - causal relation j <= i + offset when causal=True

    Raw learned kernel parameters theta are converted into a positive normalized
    kernel per batch/head:
        W = softmax(theta) over the KxK spatial support

    The refinement is:
        Y = Conv((S * V), W) / max(Conv(V, W), eps)

    with zero padding outside the block map. Invalid future/padding outputs are
    forced back to zero.

    Args:
        block_scores:
            [batch, heads, q_blocks_padded, k_blocks_padded]
        real_q_block_num / real_k_block_num:
            Number of real blocks before estimator padding. If omitted, the full
            current map extent is treated as real.
    """
    b, h, qb, kb = block_scores.shape
    device = block_scores.device
    out_dtype = block_scores.dtype

    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )

    if real_q_block_num is None:
        real_q_block_num = qb
    if real_k_block_num is None:
        real_k_block_num = kb

    # Float32 keeps the masked denominator and local renormalization stable.
    x = torch.nan_to_num(
        block_scores.float(),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    valid_base = _get_prefill_valid_block_mask_cached(
        q_blocks=qb,
        k_blocks=kb,
        real_q_blocks=real_q_block_num,
        real_k_blocks=real_k_block_num,
        device=device,
        causal=causal,
    )
    valid = valid_base.expand(b, h, qb, kb)

    # [1, B*H, Qb, Kb] for grouped conv2d.
    x = (x * valid.to(x.dtype)).reshape(1, b * h, qb, kb)
    valid_ch = valid.to(x.dtype).reshape(1, b * h, qb, kb)

    theta = _get_conv_weight(
        kernel_size=kernel_size,
        weight_path=weight_path,
        device=device,
    )
    raw_weight_cache_id = theta.data_ptr()
    theta = _select_conv_weight_for_layer_head(
        weight=theta,
        num_heads=h,
        layer_idx=layer_idx,
        batch_size=b,
        kernel_size=kernel_size,
    ).to(device=device, dtype=torch.float32)

    # Positive, per-head normalized spatial kernel:
    # W >= 0 and sum_{a,b} W[a,b] = 1.
    weight = torch.softmax(
        theta.reshape(b * h, -1),
        dim=-1,
    ).reshape(
        b * h,
        1,
        kernel_size,
        kernel_size,
    ).contiguous()

    pad = kernel_size // 2

    # Outside the map is structurally invalid, so use zero padding for both
    # signal and support mask. No replicate/reflect/circular padding.
    x_padded = F.pad(
        x,
        (pad, pad, pad, pad),
        mode="constant",
        value=0.0,
    )

    numerator = F.conv2d(
        x_padded,
        weight,
        stride=1,
        padding=0,
        groups=b * h,
    )
    support = _get_conv_support_cached(
        valid_ch=valid_ch,
        weight=weight,
        kernel_size=kernel_size,
        batch_size=b,
        num_heads=h,
        q_blocks=qb,
        k_blocks=kb,
        real_q_blocks=real_q_block_num,
        real_k_blocks=real_k_block_num,
        causal=causal,
        layer_idx=layer_idx,
        weight_path=weight_path,
        raw_weight_cache_id=raw_weight_cache_id,
        eps=eps,
    )

    smoothed = numerator / support
    smoothed = smoothed.view(b, h, qb, kb)

    # Future blocks and estimator-padding blocks stay invalid.
    smoothed = smoothed * valid.to(smoothed.dtype)

    return smoothed.to(out_dtype)

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



def _fixed_topk_mask_from_scores(
    scores: torch.Tensor,
    fixed_topk: int,
    offset: int,
    causal: bool = True,
):
    """
    Select fixed top-k key blocks for each query block from a block-score tensor.
    scores: [batch, heads, q_blocks_this_chunk, k_blocks]
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
        energy = energy.masked_fill(~causal_block[None, None, :, :], -1e9)
    else:
        causal_block = torch.ones(qb, kb, dtype=torch.bool, device=device)

    idx = torch.topk(energy, k=topk, dim=-1).indices
    mask = torch.zeros(b, h, qb, kb, dtype=torch.bool, device=device)
    mask.scatter_(-1, idx, True)

    if causal:
        mask = mask & causal_block[None, None, :, :]

    fallback_k = (torch.arange(qb, device=device) + int(offset)).clamp(0, kb - 1)
    fallback = torch.zeros(qb, kb, dtype=torch.bool, device=device)
    fallback[torch.arange(qb, device=device), fallback_k] = True
    if causal:
        fallback = fallback & causal_block

    empty = mask.sum(dim=-1) == 0
    mask = mask | (empty[:, :, :, None] & fallback[None, None, :, :])

    return mask.contiguous()



def _cheap_block_proxy_from_logits(
    attn_weights_slice: torch.Tensor,
    reshaped_block_size: int,
    mode: str = "max",
):
    """
    Build a cheap block-level proxy directly from compressed QK logits.

    Input:
        [B, H, Q_compressed, K_compressed]
    Output:
        [B, H, Q_blocks, K_blocks]

    For stride=16 and original block_size=128:
        reshaped_block_size = 8,
    so each proxy entry summarizes one 8x8 compressed region, corresponding
    to one original 128x128 token block.

    The proxy is used only for candidate preselection; it is NOT the final
    block mass and it does not apply softmax normalization.
    """
    b, h, q_len, k_len = attn_weights_slice.shape
    bs = int(reshaped_block_size)
    if q_len % bs != 0 or k_len % bs != 0:
        raise ValueError(
            f"Compressed logits shape {(q_len, k_len)} must be divisible by "
            f"reshaped_block_size={bs}"
        )

    # Future causal tiles in the upstream GEMM may be intentionally unwritten.
    # Sanitize before reduction; block-level causal masking is applied later.
    x = torch.nan_to_num(
        attn_weights_slice,
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    )
    x = x.view(
        b,
        h,
        q_len // bs,
        bs,
        k_len // bs,
        bs,
    )

    mode = str(mode).lower()
    if mode == "max":
        # Avoid materializing a float32 copy; ranking is invariant to the later
        # positive softmax scale.
        proxy = x.amax(dim=5).amax(dim=3)
    elif mode == "mean":
        proxy = x.float().mean(dim=5).mean(dim=3)
    elif mode == "sum":
        proxy = x.float().sum(dim=5).sum(dim=3)
    else:
        raise ValueError(
            f"Unsupported candidate_proxy_mode={mode!r}; "
            f"expected one of: max, mean, sum"
        )

    return proxy.contiguous()



def _topm_candidate_mask_from_proxy(
    proxy_scores: torch.Tensor,
    topm: int,
    causal: bool = True,
    keep_sink: bool = False,
    keep_recent: bool = False,
    real_q_block_num=None,
    real_k_block_num=None,
):
    """
    Per-query-block/head Top-M candidate preselection on the valid prefill
    support only.

    Estimator-padding blocks and causal-future blocks are never eligible.
    """
    b, h, qb, kb = proxy_scores.shape
    device = proxy_scores.device

    if real_q_block_num is None:
        real_q_block_num = qb
    if real_k_block_num is None:
        real_k_block_num = kb

    topm = int(topm)
    if topm <= 0:
        raise ValueError(f"candidate_topm must be positive, got {topm}")
    topm = min(topm, kb)

    valid = _build_prefill_valid_block_mask(
        q_blocks=qb,
        k_blocks=kb,
        real_q_blocks=real_q_block_num,
        real_k_blocks=real_k_block_num,
        device=device,
        causal=causal,
    ).expand(b, h, qb, kb)

    energy = torch.nan_to_num(
        proxy_scores.float(),
        nan=-1e9,
        posinf=1e9,
        neginf=-1e9,
    ).masked_fill(~valid, -1e9)

    idx = torch.topk(energy, k=topm, dim=-1).indices
    mask = torch.zeros(
        b, h, qb, kb,
        dtype=torch.bool,
        device=device,
    )
    mask.scatter_(-1, idx, True)
    mask = mask & valid

    real_q_block_num = max(0, min(int(real_q_block_num), qb))
    real_k_block_num = max(0, min(int(real_k_block_num), kb))
    offset = real_k_block_num - real_q_block_num

    # Diagonal-like fallback for real query rows only.
    q_ids = torch.arange(qb, device=device)
    fallback_k = (q_ids + int(offset)).clamp(0, max(kb - 1, 0))
    fallback = torch.zeros(
        qb, kb,
        dtype=torch.bool,
        device=device,
    )
    if kb > 0:
        fallback[q_ids, fallback_k] = True
    fallback = fallback & valid[0, 0]

    empty = mask.sum(dim=-1) == 0
    mask = mask | (
        empty[:, :, :, None]
        & fallback[None, None, :, :]
    )

    if keep_sink and real_k_block_num > 0:
        sink_valid = valid[:, :, :, 0]
        mask[:, :, :, 0] = mask[:, :, :, 0] | sink_valid

    if keep_recent:
        mask = mask | fallback[None, None, :, :]

    return (mask & valid).contiguous()


def _dilate_candidate_mask_2d(
    candidate_mask: torch.Tensor,
    kernel_size: int = 7,
    causal: bool = True,
    keep_sink: bool = False,
    keep_recent: bool = False,
    real_q_block_num=None,
    real_k_block_num=None,
):
    """
    Causal-aware 2D candidate dilation.

    Dilation uses zero outside-map padding and is intersected with the same
    structural prefill validity mask used by the masked convolution. Therefore:
        - no candidate can leak into causal-future blocks
        - no candidate can leak into estimator-padding blocks
    """
    b, h, qb, kb = candidate_mask.shape
    device = candidate_mask.device

    if real_q_block_num is None:
        real_q_block_num = qb
    if real_k_block_num is None:
        real_k_block_num = kb

    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"candidate_dilation_kernel_size must be a positive odd integer, "
            f"got {kernel_size}"
        )

    valid = _build_prefill_valid_block_mask(
        q_blocks=qb,
        k_blocks=kb,
        real_q_blocks=real_q_block_num,
        real_k_blocks=real_k_block_num,
        device=device,
        causal=causal,
    ).expand(b, h, qb, kb)

    x = (candidate_mask & valid).reshape(
        b * h, 1, qb, kb
    ).float()

    # max_pool2d padding is zero, i.e. outside-map positions do not create
    # candidates. Intersect with valid support again after dilation.
    dilated = F.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ) > 0

    dilated = dilated.view(b, h, qb, kb) & valid

    real_q_block_num = max(0, min(int(real_q_block_num), qb))
    real_k_block_num = max(0, min(int(real_k_block_num), kb))
    offset = real_k_block_num - real_q_block_num

    q_ids = torch.arange(qb, device=device)
    fallback_k = (q_ids + int(offset)).clamp(0, max(kb - 1, 0))
    fallback = torch.zeros(
        qb, kb,
        dtype=torch.bool,
        device=device,
    )
    if kb > 0:
        fallback[q_ids, fallback_k] = True
    fallback = fallback & valid[0, 0]

    empty = dilated.sum(dim=-1) == 0
    dilated = dilated | (
        empty[:, :, :, None]
        & fallback[None, None, :, :]
    )

    if keep_sink and real_k_block_num > 0:
        sink_valid = valid[:, :, :, 0]
        dilated[:, :, :, 0] = dilated[:, :, :, 0] | sink_valid

    if keep_recent:
        dilated = dilated | fallback[None, None, :, :]

    return (dilated & valid).contiguous()

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
    candidate_topm=None,
    candidate_proxy_mode="max",
    candidate_dilation_kernel_size=None,
    final_topk=None,
):
    """
    Conv block estimator with an optional hierarchical sparse block-mass path.

    Baseline path (candidate_topm is None):
        full inverse anti-diagonal estimator
        -> dense softmax_fuse_block_sum
        -> causal-aware masked normalized Conv
        -> threshold/fixed Top-K selection

    Hierarchical sparse path (candidate_topm is not None):
        full inverse anti-diagonal estimator
        -> cheap block proxy
        -> Top-M candidates
        -> 2D neighborhood dilation
        -> sparse candidate-set softmax block mass
        -> causal-aware masked normalized Conv
        -> final Top-K (or threshold fallback)

    Important:
        The inverse anti-diagonal estimator remains FULL:
            iter = 0, 1, ..., stride - 1
        No spaced anti-diagonal sampling is used.

        The sparse softmax normalizes only over the dilated candidate set, so
        this path is intentionally approximate relative to dense softmax.

    Args:
        candidate_topm:
            Enable hierarchical sparse block-mass estimation and preselect this
            many key blocks per query block/head before dilation. None disables
            the hierarchical sparse path.
        candidate_proxy_mode:
            Cheap proxy reduction over each compressed block: max/mean/sum.
        candidate_dilation_kernel_size:
            Odd 2D dilation kernel on the block grid. None uses
            conv_kernel_size (7 by default).
        final_topk:
            Alias for the final fixed Top-K after learned Conv. If both
            final_topk and fixed_topk are given, they must match.
    """
    if final_topk is not None:
        if fixed_topk is not None and int(fixed_topk) != int(final_topk):
            raise ValueError(
                f"fixed_topk={fixed_topk} and final_topk={final_topk} disagree"
            )
        fixed_topk = int(final_topk)

    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    # Real block geometry before estimator padding. These values define the
    # valid prefill support for candidate selection and masked convolution.
    real_k_block_num = (k_len + block_size - 1) // block_size
    real_q_block_num = (q_len + block_size - 1) // block_size

    k_num_to_pad = (
        ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    )
    q_num_to_pad = (
        ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    )
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    if k_num_to_pad > 0:
        pad_key_states = F.pad(
            key_states,
            (0, 0, 0, k_num_to_pad),
            value=0,
        ).to("cuda")
    else:
        pad_key_states = key_states

    if q_num_to_pad > 0:
        pad_query_states = F.pad(
            query_states,
            (0, 0, 0, q_num_to_pad),
            value=0,
        ).to("cuda")
    else:
        pad_query_states = query_states

    assert num_kv_head == num_q_head

    if use_triton and (
        "100"
        not in torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).name
    ):
        use_triton = False
        print(
            "setting use triton to false. "
            "Triton kernel not surpported on this device"
        )

    hierarchical_sparse = candidate_topm is not None
    if hierarchical_sparse and not use_triton:
        raise NotImplementedError(
            "candidate_topm hierarchical sparse block-mass path currently "
            "requires use_triton=True"
        )

    # ── Step 1: estimator geometry ──
    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = (
        reshaped_chunk_size // reshaped_block_size
    )

    # Full inverse anti-diagonal estimator.
    # The imported no-spaced-sampling kernel uses:
    #     iter = 0, 1, ..., stride - 1
    softmax_segment_size = _choose_softmax_segment_size(
        k_reshaped_seq_len=k_reshaped_seq_len,
        reshaped_block_size=reshaped_block_size,
    )

    if not use_triton:
        if select_mode == "inverse" or select_mode == "":
            reshaped_key = torch.cat(
                [
                    pad_key_states[:, :, k::stride, :]
                    for k in range(stride)
                ],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[
                        :,
                        :,
                        (stride - 1 - q) :: (stride * kdb),
                        :,
                    ]
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = torch.cat(
                [
                    pad_key_states[:, :, k::stride, :]
                    for k in range(stride)
                ],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, q::stride, :]
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "double":
            reshaped_key = torch.cat(
                [
                    pad_key_states[:, :, k::stride, :]
                    for k in range(stride)
                ],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [
                    reshaped_key[:, :, :, head_dim:],
                    reshaped_key[:, :, :, 0:head_dim],
                ],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[
                        :,
                        :,
                        (stride - 1 - q) :: stride,
                        :,
                    ]
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat(
                [
                    pad_key_states[:, :, k::stride, :]
                    for k in range(stride)
                ],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [
                    reshaped_key[:, :, :, head_dim:],
                    reshaped_key[:, :, :, 0:head_dim],
                ],
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
                    pad_query_states[
                        :,
                        :,
                        (stride - 1 - q) :: stride,
                        :,
                    ]
                    for q in range(stride)
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unsupported select_mode={select_mode!r}")

        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    # ── Step 2A: hierarchical sparse candidate block-mass path ──
    if hierarchical_sparse:
        # Keep chunk logits so that candidate construction can use a global
        # [Q_blocks, K_blocks] proxy map and 7x7 dilation can cross chunk
        # boundaries without recomputing the full anti-diagonal estimator.
        #
        # Trade-off: this uses more temporary memory than the baseline path.
        attn_weights_slice_list = []
        proxy_list = []

        for chunk_idx in range(q_chunk_num):
            if kdb != 1:
                raise ValueError(
                    "use_triton and kdb cannot be used together"
                )

            chunk_q_start = (
                chunk_idx * reshaped_chunk_size * stride
            )
            chunk_q_end = (
                (chunk_idx + 1)
                * reshaped_chunk_size
                * stride
            )

            attn_weights_slice = flat_group_gemm_fuse_reshape(
                pad_query_states[
                    :,
                    :,
                    chunk_q_start:chunk_q_end,
                    :,
                ],
                pad_key_states,
                stride,
                (
                    (k_block_num - q_block_num)
                    * reshaped_block_size
                    + chunk_idx * reshaped_chunk_size
                ),
                (
                    (k_block_num - q_block_num)
                    * reshaped_block_size
                    + chunk_idx * reshaped_chunk_size
                    + reshaped_chunk_size
                ),
                is_causal=causal,
            )

            proxy_chunk = _cheap_block_proxy_from_logits(
                attn_weights_slice,
                reshaped_block_size=reshaped_block_size,
                mode=candidate_proxy_mode,
            )

            attn_weights_slice_list.append(
                attn_weights_slice
            )
            proxy_list.append(proxy_chunk)

        if len(proxy_list) == 1:
            proxy_scores = proxy_list[0]
        else:
            proxy_scores = torch.cat(
                proxy_list,
                dim=-2,
            )

        # Top-M coarse candidates.
        candidate_mask = _topm_candidate_mask_from_proxy(
            proxy_scores,
            topm=int(candidate_topm),
            causal=causal,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
            real_q_block_num=real_q_block_num,
            real_k_block_num=real_k_block_num,
        )

        # 7x7 neighborhood dilation by default, matched to the learned Conv.
        dilation_kernel = (
            int(conv_kernel_size)
            if candidate_dilation_kernel_size is None
            else int(candidate_dilation_kernel_size)
        )
        candidate_mask = _dilate_candidate_mask_2d(
            candidate_mask,
            kernel_size=dilation_kernel,
            causal=causal,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
            real_q_block_num=real_q_block_num,
            real_k_block_num=real_k_block_num,
        )

        # Candidate-set sparse softmax block mass. Non-candidate key blocks are
        # masked at tl.load and excluded from the softmax denominator.
        attn_sum_list = []
        for chunk_idx, attn_weights_slice in enumerate(
            attn_weights_slice_list
        ):
            chunk_block_start = (
                chunk_idx * num_blocks_per_chunk
            )
            chunk_block_end = (
                (chunk_idx + 1)
                * num_blocks_per_chunk
            )
            candidate_chunk = candidate_mask[
                :,
                :,
                chunk_block_start:chunk_block_end,
                :,
            ].contiguous()

            attn_sum = sparse_softmax_fuse_block_sum(
                attn_weights_slice,
                candidate_chunk,
                reshaped_block_size,
                softmax_segment_size,
                (
                    (k_block_num - q_block_num)
                    * reshaped_block_size
                    + chunk_idx * reshaped_chunk_size
                ),
                (
                    (k_block_num - q_block_num)
                    * reshaped_block_size
                    + chunk_idx * reshaped_chunk_size
                    + reshaped_chunk_size
                ),
                k_reshaped_seq_len
                - k_reshaped_num_to_pad,
                (
                    1.4426950408889634
                    / math.sqrt(head_dim)
                    / stride
                    / norm
                ),
                is_causal=causal,
            )

            attn_sum_list.append(attn_sum)
            del attn_weights_slice

        del attn_weights_slice_list
        del proxy_list
        del proxy_scores
        del candidate_mask

    # ── Step 2B: original dense block-mass path ──
    else:
        attn_sum_list = []

        for chunk_idx in range(q_chunk_num):
            if use_triton:
                if kdb != 1:
                    raise ValueError(
                        "use_triton and kdb cannot be used together"
                    )

                attn_weights_slice = flat_group_gemm_fuse_reshape(
                    pad_query_states[
                        :,
                        :,
                        (
                            chunk_idx
                            * reshaped_chunk_size
                        )
                        * stride : (
                            chunk_idx
                            * reshaped_chunk_size
                            + reshaped_chunk_size
                        )
                        * stride,
                        :,
                    ],
                    pad_key_states,
                    stride,
                    (
                        (k_block_num - q_block_num)
                        * reshaped_block_size
                        + chunk_idx
                        * reshaped_chunk_size
                    ),
                    (
                        (k_block_num - q_block_num)
                        * reshaped_block_size
                        + chunk_idx
                        * reshaped_chunk_size
                        + reshaped_chunk_size
                    ),
                    is_causal=causal,
                )

                attn_sum = softmax_fuse_block_sum(
                    attn_weights_slice,
                    reshaped_block_size,
                    softmax_segment_size,
                    (
                        (k_block_num - q_block_num)
                        * reshaped_block_size
                        + chunk_idx
                        * reshaped_chunk_size
                    ),
                    (
                        (k_block_num - q_block_num)
                        * reshaped_block_size
                        + chunk_idx
                        * reshaped_chunk_size
                        + reshaped_chunk_size
                    ),
                    k_reshaped_seq_len
                    - k_reshaped_num_to_pad,
                    (
                        1.4426950408889634
                        / math.sqrt(head_dim)
                        / stride
                        / norm
                    ),
                    is_causal=causal,
                )

            else:
                chunked_query = reshaped_query[
                    :,
                    :,
                    (
                        chunk_idx
                        * reshaped_chunk_size
                    )
                    // kdb : (
                        chunk_idx
                        * reshaped_chunk_size
                        + reshaped_chunk_size
                    )
                    // kdb,
                    :,
                ]

                attn_weights_slice = torch.matmul(
                    chunked_query,
                    reshaped_key.transpose(2, 3),
                ).to("cuda")

                attn_weights_slice = (
                    attn_weights_slice
                    / math.sqrt(head_dim)
                    / stride
                    / norm
                )

                if causal:
                    causal_mask = torch.zeros(
                        (
                            batch_size,
                            num_q_head,
                            reshaped_chunk_size,
                            reshaped_chunk_size
                            * k_chunk_num,
                        ),
                        device=key_states.device,
                    )
                    if k_reshaped_num_to_pad > 0:
                        causal_mask[
                            :,
                            :,
                            :,
                            (-k_reshaped_num_to_pad):,
                        ] = float("-inf")

                    chunk_start = (
                        chunk_idx
                        + offset_token_chunk_num
                    ) * reshaped_chunk_size
                    chunk_end = (
                        chunk_start
                        + reshaped_chunk_size
                    )

                    causal_mask[
                        :,
                        :,
                        :,
                        chunk_start:chunk_end,
                    ] = torch.triu(
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

                    if (
                        chunk_idx == q_chunk_num - 1
                        and q_reshaped_num_to_pad != 0
                    ):
                        causal_mask[
                            :,
                            :,
                            (
                                -(
                                    q_reshaped_num_to_pad
                                    // kdb
                                )
                            ):,
                            :,
                        ] = float("-inf")

                    causal_mask[
                        :,
                        :,
                        :,
                        chunk_end:,
                    ] = float("-inf")
                    causal_mask = causal_mask[
                        :,
                        :,
                        kdb - 1 :: kdb,
                        :,
                    ]
                    attn_weights_slice = (
                        attn_weights_slice
                        + causal_mask.to(
                            attn_weights_slice.device
                        )
                    )

                if softmax:
                    attn_weights_slice = F.softmax(
                        attn_weights_slice,
                        dim=-1,
                        dtype=torch.float32,
                    ).to(pad_query_states.dtype)
                else:
                    attn_weights_slice = torch.exp(
                        attn_weights_slice
                    ).to(pad_query_states.dtype)

                attn_weights_slice = F.dropout(
                    attn_weights_slice,
                    p=0,
                    training=False,
                )

                if (
                    chunk_idx == q_chunk_num - 1
                    and q_reshaped_num_to_pad != 0
                ):
                    attn_weights_slice[
                        :,
                        :,
                        (
                            -(
                                q_reshaped_num_to_pad
                                // kdb
                            )
                        ):,
                        :,
                    ] = 0

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

    # ── Step 3: concatenate block mass and apply causal-aware masked normalized Conv ──
    if len(attn_sum_list) == 1:
        attn_sums = attn_sum_list[0]
    else:
        attn_sums = torch.cat(
            attn_sum_list,
            dim=-2,
        )

    attn_sums_smoothed = apply_conv2d_block_map(
        attn_sums,
        kernel_size=conv_kernel_size,
        weight_path=conv_weight_path,
        layer_idx=layer_idx,
        causal=causal,
        real_q_block_num=real_q_block_num,
        real_k_block_num=real_k_block_num,
    )

    # ── Step 4: final block selection ──
    attn_sums_smoothed = torch.nan_to_num(
        attn_sums_smoothed,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    # Explicit final Top-K has highest priority.
    if fixed_topk is not None:
        simple_mask_list = []
        for chunk_idx in range(q_chunk_num):
            chunk_start = (
                chunk_idx * num_blocks_per_chunk
            )
            chunk_end = (
                (chunk_idx + 1)
                * num_blocks_per_chunk
            )
            score_chunk = attn_sums_smoothed[
                :,
                :,
                chunk_start:chunk_end,
                :,
            ]
            mask_offset = (
                k_block_num
                - q_block_num
                + chunk_idx
                * num_blocks_per_chunk
            )
            simple_mask = _fixed_topk_mask_from_scores(
                score_chunk,
                fixed_topk=int(fixed_topk),
                offset=mask_offset,
                causal=causal,
            )
            simple_mask_list.append(simple_mask)

        simple_masks = torch.cat(
            simple_mask_list,
            dim=-2,
        )

    elif conv_safe_topk:
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
                chunk_start = (
                    chunk_idx * num_blocks_per_chunk
                )
                chunk_end = (
                    (chunk_idx + 1)
                    * num_blocks_per_chunk
                )

                score_chunk = attn_sums_smoothed[
                    :,
                    :,
                    chunk_start:chunk_end,
                    :,
                ]
                mask_offset = (
                    k_block_num
                    - q_block_num
                    + chunk_idx
                    * num_blocks_per_chunk
                )

                simple_mask = find_blocks_chunked(
                    score_chunk,
                    mask_offset,
                    threshold,
                    None,
                    decoding=False,
                    mode="prefill",
                    causal=causal,
                )
                simple_mask_list.append(simple_mask)

            simple_masks = torch.cat(
                simple_mask_list,
                dim=-2,
            )

        except Exception as e:
            print(
                "[Conv.py WARN] chunk-wise "
                "find_blocks_chunked failed. "
                "To use the old safe top-k fallback, "
                "rerun with --conv_safe_topk. "
                f"error={repr(e)}"
            )
            raise

    # ── Step 5: post-processing ──
    if causal:
        simple_masks[
            :,
            :,
            -q_block_num:,
            -q_block_num:,
        ] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num,
                    q_block_num,
                    dtype=bool,
                    device=simple_masks.device,
                ),
                diagonal=0,
            ),
            simple_masks[
                :,
                :,
                -q_block_num:,
                -q_block_num:,
            ],
            False,
        )

    if keep_sink:
        simple_masks[:, :, :, 0] = True

    if keep_recent:
        eye_matrix = torch.eye(
            q_block_num,
            device=simple_masks.device,
            dtype=bool,
        )
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(
                1,
                num_kv_head,
                q_block_num,
                q_block_num,
            )
        )
        simple_masks[
            :,
            :,
            -q_block_num:,
            -q_block_num:,
        ] = torch.where(
            eye_matrix_expanded,
            True,
            simple_masks[
                :,
                :,
                -q_block_num:,
                -q_block_num:,
            ],
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
    candidate_topm=None,
    candidate_proxy_mode="max",
    candidate_dilation_kernel_size=None,
    final_topk=None,
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
        candidate_topm=candidate_topm,
        candidate_proxy_mode=candidate_proxy_mode,
        candidate_dilation_kernel_size=candidate_dilation_kernel_size,
        final_topk=final_topk,
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
