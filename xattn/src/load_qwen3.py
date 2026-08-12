"""Qwen3-8B loader with xattn/flex/minference/conv/full prefill methods."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from xattn.src.Conv_qwen3 import Conv_prefill
from xattn.src.Flexprefill import Flexprefill_prefill
from xattn.src.Fullprefill import Full_prefill
from xattn.src.Minference import Minference_prefill
from xattn.src.Xattention import Xattention_prefill
from xattn.src.load_qwen import FastPrefillConfig, _maybe_unpack_density


QWEN3_NUM_LAYERS = 36
QWEN3_NUM_ATTENTION_HEADS = 32
QWEN3_NUM_KEY_VALUE_HEADS = 8


def _dense_decode(q, k, v, attention_mask, head_dim):
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : k.shape[-2]]
    probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def forward_eval(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    del output_attentions, use_cache
    if past_key_values is None:
        past_key_values = kwargs.pop("past_key_value", None)
    bsz, q_len, _ = hidden_states.shape
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads

    # Qwen3-specific: RMS-normalize Q/K before transpose and RoPE.
    query_states = self.q_norm(
        self.q_proj(hidden_states).view(bsz, q_len, num_heads, self.head_dim)
    ).transpose(1, 2)
    key_states = self.k_norm(
        self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, self.head_dim)
    ).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, num_kv_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        try:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
        except TypeError:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    config = self.fastprefillconfig

    if query_states.shape[2] == key_states.shape[2]:
        common = dict(
            stride=config.stride,
            norm=1,
            threshold=config.threshold,
            topk_ratio=config.block_topk_ratio,
        )
        density = None
        if config.metric == "xattn":
            result = Xattention_prefill(
                query_states, key_states, value_states,
                use_triton=True, return_density=config.report_density, **common,
            )
            attn_output, density = _maybe_unpack_density(result)
        elif config.metric == "conv":
            result = Conv_prefill(
                query_states, key_states, value_states,
                use_triton=config.conv_use_triton,
                conv_weight_path=config.conv_weight_path,
                layer_idx=self.layer_idx,
                conv_safe_topk=config.conv_safe_topk,
                fallback_topk=config.conv_fallback_topk,
                return_density=config.report_density,
                **common,
            )
            attn_output, density = _maybe_unpack_density(result)
        elif config.metric == "flex":
            attn_output = Flexprefill_prefill(
                query_states.transpose(1, 2), key_states.transpose(1, 2),
                value_states.transpose(1, 2), topk_ratio=config.block_topk_ratio,
            ).transpose(1, 2)
        elif config.metric == "minference":
            attn_output = Minference_prefill(query_states, key_states, value_states)
        elif config.metric == "full":
            attn_output = Full_prefill(
                query_states, key_states, value_states, attention_mask=attention_mask
            )
        else:
            raise ValueError(f"Unknown prefill metric: {config.metric}")
        if config.report_density and density is not None:
            config.add_density_record(config.metric, self.layer_idx, density)
    else:
        attn_output = _dense_decode(
            query_states, key_states, value_states, attention_mask, self.head_dim
        )

    expected = (bsz, num_heads, q_len, self.head_dim)
    if tuple(attn_output.shape) != expected:
        raise ValueError(f"attn_output must have shape {expected}, got {tuple(attn_output.shape)}")
    attn_output = self.o_proj(
        attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    )
    return attn_output, None


def load_model(fastprefillconfig: Optional[FastPrefillConfig] = None, name_or_path: str = ""):
    if fastprefillconfig is None:
        fastprefillconfig = FastPrefillConfig()
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        trust_remote_code=True,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    expected = ("qwen3", QWEN3_NUM_LAYERS, QWEN3_NUM_ATTENTION_HEADS, QWEN3_NUM_KEY_VALUE_HEADS)
    actual = (
        getattr(model.config, "model_type", None),
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
    )
    if actual != expected:
        raise ValueError(f"This loader targets Qwen3-8B {expected}, got {actual}")
    for layer in model.model.layers:
        attention = layer.self_attn
        attention.num_heads = model.config.num_attention_heads
        attention.num_key_value_heads = model.config.num_key_value_heads
        attention.num_key_value_groups = num_groups = (
            model.config.num_attention_heads // model.config.num_key_value_heads
        )
        del num_groups
        attention.fastprefillconfig = fastprefillconfig
        attention.forward = forward_eval.__get__(attention)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
