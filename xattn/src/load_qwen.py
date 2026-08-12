"""Qwen2.5-7B-Instruct loader with the project's custom prefill methods.

This is intentionally separate from ``load_llama.py`` so the existing Llama
path remains untouched.  It targets the Qwen2 attention API used by the
Transformers version this repository's Llama monkey patch was written for.
"""

from __future__ import annotations

import inspect
import math
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, StaticCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from xattn.src.Conv_qwen import Conv_prefill
from xattn.src.Flexprefill import Flexprefill_prefill
from xattn.src.Fullprefill import Full_prefill
from xattn.src.Minference import Minference_prefill
from xattn.src.Xattention import Xattention_prefill


QWEN_NUM_LAYERS = 28
QWEN_NUM_ATTENTION_HEADS = 28


def _maybe_unpack_density(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


class FastPrefillConfig(dict):
    """Qwen-specific counterpart of ``load_llama.FastPrefillConfig``."""

    def __init__(
        self,
        threshold: Optional[float] = None,
        print_detail: bool = False,
        stride: int = 16,
        metric: str = "xattn",
        conv_weight_path: Optional[str] = None,
        conv_safe_topk: bool = False,
        conv_use_triton: bool = True,
        conv_fallback_topk: int = 8,
        block_topk_ratio: Optional[float] = None,
        report_density: bool = False,
        print_density_per_layer: bool = False,
    ):
        super().__init__()
        if block_topk_ratio is not None:
            block_topk_ratio = float(block_topk_ratio)
            if not 0.0 < block_topk_ratio <= 1.0:
                raise ValueError(
                    "block_topk_ratio must be in (0, 1], "
                    f"got {block_topk_ratio}"
                )

        self.threshold = float(0.9 if threshold is None else threshold)
        self.print_detail = bool(print_detail)
        self.stride = int(stride)
        self.metric = str(metric)
        # Retained for API compatibility. Conv_qwen deliberately keeps the PT
        # path hard-coded in the existing Conv.py implementation.
        self.conv_weight_path = conv_weight_path
        self.conv_safe_topk = bool(conv_safe_topk)
        self.conv_use_triton = bool(conv_use_triton)
        self.conv_fallback_topk = int(conv_fallback_topk)
        self.block_topk_ratio = block_topk_ratio
        self.report_density = bool(report_density)
        self.print_density_per_layer = bool(print_density_per_layer)
        self.density_records = []

    def reset_density_records(self):
        self.density_records = []

    def add_density_record(self, metric: str, layer_idx: int, density):
        try:
            density_value = float(density)
        except (TypeError, ValueError):
            density_value = float("nan")
        record = {
            "metric": str(metric),
            "layer_idx": int(layer_idx),
            "density": density_value,
        }
        self.density_records.append(record)
        if self.print_density_per_layer:
            print(
                f"[Density] metric={record['metric']} layer={record['layer_idx']} "
                f"density={record['density']:.6f}",
                flush=True,
            )

    def get_density_summary(self):
        grouped = {}
        for record in self.density_records:
            density = record["density"]
            if math.isnan(density):
                continue
            metric = record["metric"]
            grouped.setdefault(metric, []).append(density)
        return {
            metric: sum(values) / len(values)
            for metric, values in grouped.items()
            if values
        }

    def print_density_summary(self):
        summary = self.get_density_summary()
        if not summary:
            print("[Density] no density records collected")
            return
        values = " ".join(f"{key}={value:.6f}" for key, value in summary.items())
        print(f"[Density Summary] {values}", flush=True)


def _dense_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_dim: int,
):
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    return torch.matmul(probs, value_states)


def forward_eval(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    # Transformers' newer Qwen2 API renamed this argument to the plural form
    # and changed attention's return tuple from three values to two.
    if past_key_value is None:
        past_key_value = kwargs.pop("past_key_values", None)
    bsz, q_len, _ = hidden_states.size()
    num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
    num_kv_heads = getattr(
        self, "num_key_value_heads", self.config.num_key_value_heads
    )

    query_states = self.q_proj(hidden_states).view(
        bsz, q_len, num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(
        bsz, q_len, num_kv_heads, self.head_dim
    ).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, num_kv_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    if isinstance(past_key_value, StaticCache) and cache_position is not None:
        cache_length = min(int(cache_position[-1]) + 1, key_states.shape[2])
        key_states = key_states[:, :, :cache_length, :]
        value_states = value_states[:, :, :cache_length, :]

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    is_prefill = query_states.shape[2] == key_states.shape[2]
    config = self.fastprefillconfig

    if is_prefill:
        common = dict(
            stride=config.stride,
            norm=1,
            threshold=config.threshold,
            topk_ratio=config.block_topk_ratio,
        )
        if config.metric == "xattn":
            result = Xattention_prefill(
                query_states,
                key_states,
                value_states,
                use_triton=True,
                return_density=config.report_density,
                **common,
            )
            attn_output, density = _maybe_unpack_density(result)
        elif config.metric == "conv":
            result = Conv_prefill(
                query_states,
                key_states,
                value_states,
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
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                topk_ratio=config.block_topk_ratio,
            ).transpose(1, 2)
            density = None
        elif config.metric == "minference":
            attn_output = Minference_prefill(query_states, key_states, value_states)
            density = None
        elif config.metric == "full":
            attn_output = Full_prefill(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
            )
            density = None
        else:
            raise ValueError(f"Unknown prefill metric: {config.metric}")

        if config.report_density and density is not None:
            config.add_density_record(config.metric, self.layer_idx, density)
    else:
        attn_output = _dense_decode_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.head_dim,
        )

    expected = (bsz, num_heads, q_len, self.head_dim)
    if attn_output.size() != expected:
        raise ValueError(
            f"attn_output should have shape {expected}, got {tuple(attn_output.size())}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    attn_weights = None
    if getattr(self, "_fastprefill_returns_past_key_value", True):
        return attn_output, attn_weights, past_key_value
    return attn_output, attn_weights


def load_model(
    fastprefillconfig: Optional[FastPrefillConfig] = None,
    name_or_path: str = "",
):
    """Load Qwen2.5 and install custom prefill on every self-attention layer."""
    if fastprefillconfig is None:
        fastprefillconfig = FastPrefillConfig()

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        trust_remote_code=True,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()

    if getattr(model.config, "model_type", None) != "qwen2":
        raise ValueError(
            f"load_qwen expected model_type='qwen2', got {model.config.model_type!r}"
        )

    if model.config.num_hidden_layers != QWEN_NUM_LAYERS:
        raise ValueError(
            "This adapter targets Qwen2.5-7B-Instruct with 28 layers; "
            f"got {model.config.num_hidden_layers}."
        )
    if model.config.num_attention_heads != QWEN_NUM_ATTENTION_HEADS:
        raise ValueError(
            "This adapter targets Qwen2.5-7B-Instruct with 28 attention heads; "
            f"got {model.config.num_attention_heads}."
        )

    for layer in model.model.layers:
        attention = layer.self_attn
        original_parameters = inspect.signature(attention.forward).parameters
        attention._fastprefill_returns_past_key_value = (
            "output_attentions" in original_parameters
            and "past_key_value" in original_parameters
        )
        # Transformers 4.45 exposes these directly. Set them explicitly for
        # nearby compatible versions that only keep them in config.
        attention.num_heads = model.config.num_attention_heads
        attention.num_key_value_heads = model.config.num_key_value_heads
        attention.num_key_value_groups = (
            model.config.num_attention_heads // model.config.num_key_value_heads
        )
        attention.fastprefillconfig = fastprefillconfig
        attention.forward = forward_eval.__get__(attention)

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
