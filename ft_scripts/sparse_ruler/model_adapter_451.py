"""Frozen-teacher model helpers for Transformers 4.51.0.

Llama and Qwen3 expose nearly identical decoder stacks in 4.51, but Qwen3
normalizes Q/K after projection and before RoPE.  Training must mirror that
detail or the learned Conv selector sees a score map different from inference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers.utils.versions import require_version


require_version(
    "transformers==4.51.0",
    "Conv-kernel training targets the Transformers 4.51.0 model interfaces.",
)

SUPPORTED_LAYOUTS = {
    "llama": (32, 32, 8),
    "qwen3": (36, 32, 8),
}


def validate_model_451(
    model,
    *,
    expected_model_type: str,
    expected_num_layers: Optional[int] = None,
    expected_num_heads: Optional[int] = None,
    expected_num_key_value_heads: Optional[int] = None,
) -> None:
    model_type = str(getattr(model.config, "model_type", ""))
    if expected_model_type not in SUPPORTED_LAYOUTS:
        raise ValueError(f"unsupported expected model type: {expected_model_type!r}")
    if model_type != expected_model_type:
        raise ValueError(
            f"expected model_type={expected_model_type!r}, got {model_type!r}"
        )

    actual = (
        int(model.config.num_hidden_layers),
        int(model.config.num_attention_heads),
        int(model.config.num_key_value_heads),
    )
    expected = (
        int(expected_num_layers or SUPPORTED_LAYOUTS[model_type][0]),
        int(expected_num_heads or SUPPORTED_LAYOUTS[model_type][1]),
        int(
            expected_num_key_value_heads
            or SUPPORTED_LAYOUTS[model_type][2]
        ),
    )
    if actual != expected:
        raise ValueError(
            "model/checkpoint layout does not match training arguments: "
            f"expected layers/q_heads/kv_heads={expected}, got {actual}"
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        (q * cos) + (_rotate_half(q) * sin),
        (k * cos) + (_rotate_half(k) * sin),
    )


@torch.inference_mode()
def extract_qk_451(
    model,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    attention = model.model.layers[layer_idx].self_attn
    batch, seq_len, _ = hidden_states.shape
    q_heads = int(model.config.num_attention_heads)
    kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(
        getattr(
            attention,
            "head_dim",
            model.config.hidden_size // q_heads,
        )
    )

    query_states = attention.q_proj(hidden_states).view(
        batch, seq_len, q_heads, head_dim
    )
    key_states = attention.k_proj(hidden_states).view(
        batch, seq_len, kv_heads, head_dim
    )
    if model.config.model_type == "qwen3":
        query_states = attention.q_norm(query_states)
        key_states = attention.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    cos, sin = model.model.rotary_emb(hidden_states, position_ids)
    query_states, key_states = _apply_rope(
        query_states,
        key_states,
        cos,
        sin,
    )
    return query_states.contiguous(), key_states.contiguous()
