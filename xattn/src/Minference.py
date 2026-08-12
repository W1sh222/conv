import math

import torch

from minference.ops.pit_sparse_flash_attention_v2 import (
    vertical_slash_sparse_attention,
)


# ============================================================
# MInference-VS constants
# ============================================================

MAX_LAST_Q = 64

_arange = torch.arange(MAX_LAST_Q, device="cuda")
LAST_Q_MASK = (
    _arange[None, None, :, None]
    >= _arange[None, None, None, :]
)


def sum_all_diagonal_matrix(
    mat: torch.Tensor,
) -> torch.Tensor:
    """
    Sum attention scores along slash diagonals.

    Args:
        mat:
            Tensor with shape [1, 1, n, m].

    Returns:
        Tensor with shape [1, 1, n + m - 1].

    Notes:
        This helper follows the original MInference-VS
        single-head reference implementation.
    """
    b, h, n, m = mat.shape

    if b != 1 or h != 1:
        raise ValueError(
            "sum_all_diagonal_matrix expects single-head input "
            f"[1, 1, n, m], got {mat.shape}"
        )

    zero_mat = torch.zeros(
        (b, h, n, n),
        dtype=mat.dtype,
        device=mat.device,
    )

    mat_padded = torch.cat(
        (zero_mat, mat, zero_mat),
        dim=-1,
    )

    mat_strided = mat_padded.as_strided(
        size=(1, 1, n, n + m),
        stride=(
            1,
            n * (2 * n + m),
            2 * n + m + 1,
            1,
        ),
    )

    sum_diags = torch.sum(
        mat_strided,
        dim=2,
    )

    return sum_diags[:, :, 1:]


@torch.no_grad()
def Minference_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    vertical_size: int = 500,
    slash_size: int = 3048,
) -> torch.Tensor:
    """
    MInference-VS sparse prefill.

    This implementation uses a fixed Vertical-Slash pattern budget:

        vertical_size = 1000
        slash_size    = 6096

    by default.

    Args:
        query_states:
            [batch, heads, seq_len, head_dim]

        key_states:
            [batch, heads, seq_len, head_dim]

        value_states:
            [batch, heads, seq_len, head_dim]

        vertical_size:
            Number of selected vertical indices.

        slash_size:
            Number of selected slash/diagonal indices.

    Returns:
        Tensor with shape:
            [batch, heads, seq_len, head_dim]
    """

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    if query_states.ndim != 4:
        raise ValueError(
            "query_states must have shape [B, H, N, D], "
            f"got {query_states.shape}"
        )

    if query_states.shape != key_states.shape:
        raise ValueError(
            "query_states and key_states must have the same shape, "
            f"got {query_states.shape} and {key_states.shape}"
        )

    if key_states.shape != value_states.shape:
        raise ValueError(
            "key_states and value_states must have the same shape, "
            f"got {key_states.shape} and {value_states.shape}"
        )

    device = query_states.device

    # .to() is not in-place.
    key_states = key_states.to(device)
    value_states = value_states.to(device)

    batch_size, num_heads, q_len, head_dim = (
        query_states.shape
    )

    if batch_size != 1:
        raise ValueError(
            "This MInference-VS reference implementation "
            f"currently expects batch_size=1, got {batch_size}"
        )

    if q_len <= 0:
        raise ValueError(
            f"q_len must be positive, got {q_len}"
        )

    # --------------------------------------------------------
    # Fixed VS budget.
    #
    # Match MInference behavior:
    # vertical >= 30
    # slash    >= 50
    # and neither can exceed sequence length.
    # --------------------------------------------------------

    vertical_topk_size = min(
        q_len,
        max(int(vertical_size), 30),
    )

    slash_topk_size = min(
        q_len,
        max(int(slash_size), 50),
    )

    output = torch.empty_like(query_states)

    # ========================================================
    # Per-head MInference-VS
    # ========================================================

    for head in range(num_heads):

        q = query_states[
            :, head : head + 1, :, :
        ]

        k = key_states[
            :, head : head + 1, :, :
        ]

        v = value_states[
            :, head : head + 1, :, :
        ]

        # ----------------------------------------------------
        # Use last 64 queries to estimate sparse pattern.
        # ----------------------------------------------------

        last_q = min(
            MAX_LAST_Q,
            q_len,
        )

        # qk:
        # [B, 1, last_q, q_len]
        qk = torch.einsum(
            "bhmd,bhnd->bhmn",
            q[:, :, -last_q:, :],
            k,
        )

        # Important:
        # use actual head_dim instead of hard-coded sqrt(128).
        qk = qk / math.sqrt(head_dim)

        # ----------------------------------------------------
        # Causal mask for sampled tail queries.
        # ----------------------------------------------------

        causal_mask = LAST_Q_MASK[
            ...,
            -last_q:,
            -last_q:,
        ].to(device)

        qk[:, :, :, -last_q:] = torch.where(
            causal_mask,
            qk[:, :, :, -last_q:],
            -torch.inf,
        )

        # MInference computes pattern scores using FP32 softmax.
        qk = torch.softmax(
            qk,
            dim=-1,
            dtype=torch.float32,
        )

        # ====================================================
        # Vertical selection
        # ====================================================

        vertical_scores = qk.sum(
            dim=-2,
            keepdim=True,
        )

        # Preserve sink tokens.
        sink_count = min(
            30,
            vertical_topk_size,
            q_len,
        )

        if sink_count > 0:
            vertical_scores[
                ..., :sink_count
            ] = torch.inf

        vertical_topk = torch.topk(
            vertical_scores,
            k=vertical_topk_size,
            dim=-1,
        ).indices

        # ====================================================
        # Slash selection
        # ====================================================

        diagonal_scores = (
            sum_all_diagonal_matrix(qk)
        )

        # Match original MInference-VS indexing:
        #
        # for normal q_len > 1:
        #     [..., :-last_q + 1]
        #
        # gives q_len candidate slash patterns.
        if last_q > 1:
            slash_scores = diagonal_scores[
                ..., : -last_q + 1
            ]
        else:
            slash_scores = diagonal_scores[
                ..., :q_len
            ]

        # Preserve recent/local slash patterns.
        recent_count = min(
            100,
            slash_topk_size,
            slash_scores.shape[-1],
        )

        if recent_count > 0:
            slash_scores[
                ..., -recent_count:
            ] = torch.inf

        slash_score_indices = torch.topk(
            slash_scores,
            k=slash_topk_size,
            dim=-1,
        ).indices

        # Convert score positions to the slash-index
        # convention required by MInference kernel.
        slash_indices = (
            q_len - 1
        ) - slash_score_indices

        # ====================================================
        # Sparse attention kernel
        # ====================================================

        output[
            :, head : head + 1, :, :
        ] = vertical_slash_sparse_attention(
            q,
            k,
            v,
            vertical_topk,
            slash_indices,
        )

    return output