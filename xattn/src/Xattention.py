from xattn.src.utils import *
import torch
import math
import torch.nn.functional as F
from xattn.src.kernels import (
    flat_group_gemm,
    softmax_fuse_block_sum,
    flat_group_gemm_fuse_reshape,
)
from block_sparse_attn import block_sparse_attn_func


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
    Build a hard mask by selecting exactly the same per-query top-k budget from
    a block-score tensor.

    scores: [batch, heads, q_blocks_this_chunk, k_blocks]
    offset: same offset passed to find_blocks_chunked.

    This is for fair timing under a fixed top-k budget. It replaces the
    threshold/top-p style find_blocks_chunked selection, but keeps the score
    computation unchanged.
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

    # 保底：每个 query block 至少保留当前 causal 对角 block，避免全 False 行。
    fallback_k = (torch.arange(qb, device=device) + int(offset)).clamp(0, kb - 1)
    fallback = torch.zeros(qb, kb, dtype=torch.bool, device=device)
    fallback[torch.arange(qb, device=device), fallback_k] = True
    if causal:
        fallback = fallback & causal_block
    empty = mask.sum(dim=-1) == 0
    mask = mask | (empty[:, :, :, None] & fallback[None, None, :, :])

    return mask.contiguous()



def _topk_ratio_mask_from_scores(
    scores: torch.Tensor,
    topk_ratio: float,
    offset: int,
    causal: bool = True,
):
    """
    Select a fixed ratio of each query block's valid key-block support.

    For causal prefill, local query row i keeps:

        K_i = ceil(topk_ratio * visible_key_blocks_i)

    where:
        visible_key_blocks_i = clamp(i + offset + 1, 0, k_blocks)

    This is different from selecting topk_ratio * padded_k_blocks globally.
    Each query uses its own causal-visible support.

    Args:
        scores:
            [batch, heads, q_blocks_this_chunk, k_blocks]
        topk_ratio:
            Float in (0, 1].
        offset:
            Global query/key block offset for this query chunk.
    """
    b, h, qb, kb = scores.shape
    device = scores.device

    ratio = float(topk_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(
            f"topk_ratio must be in (0, 1], got {topk_ratio}"
        )

    if qb == 0 or kb == 0:
        return torch.zeros(
            b, h, qb, kb,
            dtype=torch.bool,
            device=device,
        )

    energy = torch.nan_to_num(
        scores.float(),
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    )

    q_idx = torch.arange(qb, device=device)
    k_idx = torch.arange(kb, device=device)

    if causal:
        causal_block = (
            k_idx[None, :]
            <= (q_idx[:, None] + int(offset))
        )
        visible_counts = (
            q_idx + int(offset) + 1
        ).clamp(min=0, max=kb)

        energy = energy.masked_fill(
            ~causal_block[None, None, :, :],
            -1e9,
        )

        # Maximum valid support in this local query chunk.
        max_visible = max(
            0,
            min(kb, qb + int(offset)),
        )
    else:
        causal_block = torch.ones(
            qb,
            kb,
            dtype=torch.bool,
            device=device,
        )
        visible_counts = torch.full(
            (qb,),
            kb,
            dtype=torch.long,
            device=device,
        )
        max_visible = kb

    if max_visible <= 0:
        return torch.zeros(
            b, h, qb, kb,
            dtype=torch.bool,
            device=device,
        )

    # Per-query variable K.
    k_per_q = torch.ceil(
        visible_counts.to(torch.float32) * ratio
    ).to(torch.long)
    k_per_q = torch.where(
        visible_counts > 0,
        k_per_q.clamp(min=1),
        torch.zeros_like(k_per_q),
    )

    # torch.topk accepts one scalar K. Select the largest K needed in this
    # chunk, then keep only the first k_per_q entries for each query row.
    max_topk = max(
        1,
        min(kb, int(math.ceil(ratio * max_visible))),
    )

    idx = torch.topk(
        energy,
        k=max_topk,
        dim=-1,
    ).indices

    rank = torch.arange(
        max_topk,
        device=device,
    )[None, None, None, :]
    keep_rank = rank < k_per_q[None, None, :, None]
    keep_rank = keep_rank.expand(b, h, qb, max_topk)

    mask = torch.zeros(
        b,
        h,
        qb,
        kb,
        dtype=torch.bool,
        device=device,
    )
    mask.scatter_(
        dim=-1,
        index=idx,
        src=keep_rank,
    )

    mask = mask & causal_block[None, None, :, :]

    # Safety fallback: every row with legal causal support keeps its most
    # recent legal key block.
    fallback_k = (
        q_idx + int(offset)
    ).clamp(0, kb - 1)
    fallback = torch.zeros(
        qb,
        kb,
        dtype=torch.bool,
        device=device,
    )
    fallback[q_idx, fallback_k] = True
    fallback = fallback & causal_block

    empty = mask.sum(dim=-1) == 0
    mask = mask | (
        empty[:, :, :, None]
        & fallback[None, None, :, :]
    )

    return mask.contiguous()



def xattn_estimate(
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
    fixed_topk=None,
    topk_ratio=None,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    if topk_ratio is not None:
        topk_ratio = float(topk_ratio)
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError(
                f"topk_ratio must be in (0, 1], got {topk_ratio}"
            )
        if fixed_topk is not None:
            raise ValueError(
                "Use only one XAttention block budget: "
                "topk_ratio or fixed_topk, not both."
            )

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
    attn_sum_list = []
    simple_mask_list = []

    if use_triton and (
        "100" not in torch.cuda.get_device_properties(torch.cuda.current_device()).name
    ):
        use_triton = False
        print(
            "setting use triton to false. Triton kernel not surpported on this device"
        )

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    if not use_triton:
        if select_mode == "random":
            perm_idx = torch.randperm(stride)
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, perm_idx[i] :: stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "inverse" or select_mode == "":
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
                [reshaped_key[:, :, :, -head_dim:], reshaped_key[:, :, :, 0:-head_dim]],
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
                if k_reshaped_num_to_pad > 0:
                    causal_mask[
                        :, :, :, -k_reshaped_num_to_pad:
                    ] = float("-inf")
                chunk_start = (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
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
        
        mask_offset = (
            k_block_num
            - q_block_num
            + chunk_idx * num_blocks_per_chunk
        )

        if topk_ratio is not None:
            simple_mask = _topk_ratio_mask_from_scores(
                attn_sum,
                topk_ratio=topk_ratio,
                offset=mask_offset,
                causal=causal,
            )
        elif fixed_topk is not None:
            simple_mask = _fixed_topk_mask_from_scores(
                attn_sum,
                fixed_topk=int(fixed_topk),
                offset=mask_offset,
                causal=causal,
            )
        else:
            simple_mask = find_blocks_chunked(
                attn_sum,
                mask_offset,
                threshold,
                None,
                decoding=False,
                mode="prefill",
                causal=causal,
            )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    if not use_triton:
        del reshaped_query, reshaped_key
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

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
        # Sink means key block 0 for every query block.
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


def Xattention_prefill(
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
    return_density=False,
    fixed_topk=None,
    topk_ratio=None,
):
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
    attn_sums, approx_simple_mask = xattn_estimate(
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
        fixed_topk=fixed_topk,
        topk_ratio=topk_ratio,
    )

    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    ####################
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

    sparse_mask = approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous()
    average_density = None
    if return_density:
        average_density = _compute_average_density_from_mask(
            sparse_mask,
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
        sparse_mask,
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )
    attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(
        1, 2
    )
    ################################

    del query_states
    num_to_compute = (k_block_num + 1) * k_block_num / 2 * num_heads
    
    # selected_blocks = approx_simple_mask.sum().item()

    # num_to_compute = (
    #     (k_block_num + 1)
    #     * k_block_num
    #     / 2
    #     * num_heads
    # )

    # density = selected_blocks / num_to_compute

    # print(
    #     f"[XAttn] "
    #     f"selected={selected_blocks:.0f} "
    #     f"total={num_to_compute:.0f} "
    #     f"density={density*100:.2f}%"
    # )
    # print(f"approximated prefilling Computation: {approx_simple_mask.sum() / num_to_compute}")
    del approx_simple_mask, attn_sums, sparse_mask
    if return_density:
        return attn_output, average_density
    return attn_output