"""Shared Transformers 4.51.0 sparse-prefill attention adapter.

The upstream Llama and Qwen3 attention modules have the same 4.51 return
contract, but Qwen3 additionally applies per-head Q/K RMSNorm. Keeping the
method dispatch here prevents the two model loaders from drifting apart.
"""

from __future__ import annotations

import math
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, StaticCache
from transformers.utils.versions import require_version


require_version(
    "transformers==4.51.0",
    "This repository's attention monkey patches target Transformers 4.51.0.",
)

SUPPORTED_METHODS = {"xattn", "conv", "minference", "flex", "full"}


def _maybe_unpack_density(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, float("nan")


class BaseFastPrefillConfig(dict):
    """Device-agnostic configuration shared by the Llama and Qwen3 loaders."""

    def __init__(
        self,
        *,
        threshold: Any = 0.9,
        print_detail: bool = False,
        stride: int = 16,
        metric: str = "xattn",
        conv_weight_path: Optional[str] = None,
        default_conv_weight_path: Optional[str] = None,
        conv_safe_topk: bool = False,
        conv_use_triton: bool = True,
        conv_fallback_topk: int = 8,
        block_topk_ratio: Optional[float] = None,
        report_density: bool = False,
        print_density_per_layer: bool = False,
        rope_scaling_type: str = "none",
        rope_factor: float = 4.0,
        rope_original_max_position_embeddings: int = 32768,
        max_position_embeddings_override: Optional[int] = None,
    ):
        super().__init__()
        metric = str(metric).lower()
        if metric not in SUPPORTED_METHODS:
            raise ValueError(
                f"metric must be one of {sorted(SUPPORTED_METHODS)}, got {metric!r}"
            )
        if int(stride) <= 0:
            raise ValueError("stride must be positive")
        if block_topk_ratio is not None:
            block_topk_ratio = float(block_topk_ratio)
            if not 0.0 < block_topk_ratio <= 1.0:
                raise ValueError("block_topk_ratio must be in (0, 1]")
        rope_scaling_type = str(rope_scaling_type).lower()
        if rope_scaling_type not in {"none", "yarn"}:
            raise ValueError("rope_scaling_type must be 'none' or 'yarn'")
        if float(rope_factor) <= 0.0:
            raise ValueError("rope_factor must be positive")
        if int(rope_original_max_position_embeddings) <= 0:
            raise ValueError("rope_original_max_position_embeddings must be positive")
        if (
            max_position_embeddings_override is not None
            and int(max_position_embeddings_override) <= 0
        ):
            raise ValueError("max_position_embeddings_override must be positive")

        self.threshold = threshold
        self.print_detail = bool(print_detail)
        self.stride = int(stride)
        self.metric = metric
        self.conv_weight_path = conv_weight_path or default_conv_weight_path
        self.conv_safe_topk = bool(conv_safe_topk)
        self.conv_use_triton = bool(conv_use_triton)
        self.conv_fallback_topk = int(conv_fallback_topk)
        self.block_topk_ratio = block_topk_ratio
        self.report_density = bool(report_density)
        self.print_density_per_layer = bool(print_density_per_layer)
        self.rope_scaling_type = rope_scaling_type
        self.rope_factor = float(rope_factor)
        self.rope_original_max_position_embeddings = int(
            rope_original_max_position_embeddings
        )
        self.max_position_embeddings_override = (
            None
            if max_position_embeddings_override is None
            else int(max_position_embeddings_override)
        )
        self.density_records = []

    def threshold_for_layer(self, layer_idx: int, device: torch.device):
        threshold = self.threshold
        if not isinstance(threshold, torch.Tensor):
            return threshold
        if threshold.ndim >= 2:
            if not 0 <= layer_idx < threshold.shape[0]:
                raise IndexError(
                    f"threshold has {threshold.shape[0]} layers, got layer {layer_idx}"
                )
            threshold = threshold[layer_idx]
        return threshold.to(device=device)

    def reset_density_records(self):
        self.density_records = []

    def add_density_record(self, metric: str, layer_idx: int, density):
        try:
            if isinstance(density, torch.Tensor):
                density = density.detach().float().mean().cpu().item()
            value = float(density)
        except (TypeError, ValueError):
            value = float("nan")
        record = {
            "metric": str(metric),
            "layer_idx": int(layer_idx),
            "density": value,
        }
        self.density_records.append(record)
        if self.print_density_per_layer:
            print(
                f"[Density] metric={metric} layer={layer_idx} density={value:.6f}",
                flush=True,
            )

    def get_density_summary(self):
        grouped = {}
        for record in self.density_records:
            value = record["density"]
            if math.isnan(value):
                continue
            grouped.setdefault(record["metric"], []).append(value)
        return {
            metric: sum(values) / len(values)
            for metric, values in grouped.items()
            if values
        }

    def print_density_summary(self):
        summary = self.get_density_summary()
        if not summary:
            print("[Density] no density records collected", flush=True)
            return
        values = " ".join(f"{key}={value:.6f}" for key, value in summary.items())
        print(f"[Density Summary] {values}", flush=True)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(
        batch, kv_heads, repetitions, seq_len, head_dim
    )
    return expanded.reshape(batch, kv_heads * repetitions, seq_len, head_dim)


def _crop_static_cache(key_states, value_states, cache, cache_position, layer_idx):
    if not isinstance(cache, StaticCache):
        return key_states, value_states
    if cache_position is not None and cache_position.numel():
        cache_length = int(cache_position[-1].item()) + 1
    else:
        cache_length = int(cache.get_seq_length(layer_idx))
    cache_length = min(cache_length, key_states.shape[-2])
    return key_states[..., :cache_length, :], value_states[..., :cache_length, :]


def _dense_attention(query, key, value, attention_mask):
    q_len = query.shape[-2]
    k_len = key.shape[-2]
    mask = attention_mask
    if isinstance(mask, torch.Tensor):
        mask = mask[..., -q_len:, :k_len].to(query.device)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=mask is None and q_len == k_len,
    )


def _capture_qk_if_requested(self, query_states, key_states):
    if self.layer_idx != getattr(self, "layer_to_save", -1):
        return
    target_len = int(getattr(self, "target_len"))
    output_dir = Path(getattr(self, "capture_output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / f"query_{target_len}.pkl"
    key_path = output_dir / f"key_{target_len}.pkl"
    query_cpu = query_states.detach().cpu()
    if query_path.exists() and query_path.stat().st_size:
        with query_path.open("rb") as handle:
            old_query = pickle.load(handle)
        query_cpu = torch.cat((old_query, query_cpu), dim=-2)
    with query_path.open("wb") as handle:
        pickle.dump(query_cpu, handle)
    if key_states.shape[-2] == target_len:
        with key_path.open("wb") as handle:
            pickle.dump(key_states.detach().cpu(), handle)


def _run_prefill(self, query_states, key_states, value_states, attention_mask):
    config: BaseFastPrefillConfig = self.fastprefillconfig
    method = config.metric
    threshold = config.threshold_for_layer(self.layer_idx, query_states.device)

    if method == "xattn":
        from xattn.src.Xattention import Xattention_prefill

        result = Xattention_prefill(
            query_states,
            key_states,
            value_states,
            stride=config.stride,
            norm=1,
            threshold=threshold,
            use_triton=True,
            topk_ratio=config.block_topk_ratio,
            return_density=config.report_density,
        )
    elif method == "conv":
        if self.config.model_type == "qwen3":
            from xattn.src.Conv_qwen3 import Conv_prefill
        else:
            from xattn.src.Conv import Conv_prefill

        result = Conv_prefill(
            query_states,
            key_states,
            value_states,
            stride=config.stride,
            norm=1,
            threshold=threshold,
            use_triton=config.conv_use_triton,
            conv_weight_path=config.conv_weight_path,
            layer_idx=self.layer_idx,
            conv_safe_topk=config.conv_safe_topk,
            fallback_topk=config.conv_fallback_topk,
            topk_ratio=config.block_topk_ratio,
            return_density=config.report_density,
        )
    elif method == "flex":
        from xattn.src.Flexprefill import Flexprefill_prefill

        output = Flexprefill_prefill(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            topk_ratio=config.block_topk_ratio,
        )
        return output.transpose(1, 2)
    elif method == "minference":
        from xattn.src.Minference import Minference_prefill

        return Minference_prefill(query_states, key_states, value_states)
    elif method == "full":
        return _dense_attention(query_states, key_states, value_states, attention_mask)
    else:
        raise ValueError(f"Unsupported metric: {method}")

    output, density = _maybe_unpack_density(result)
    if config.report_density:
        config.add_density_record(method, self.layer_idx, density)
    return output


@torch.no_grad()
def forward_eval_451(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Transformers 4.51 attention contract: return ``(output, weights)``."""
    del kwargs
    started = time.perf_counter() if self.fastprefillconfig.print_detail else None
    batch, q_len, _ = hidden_states.shape
    num_heads = self.config.num_attention_heads
    head_dim = self.head_dim
    hidden_shape = (batch, q_len, -1, head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)
    if self.config.model_type == "qwen3":
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if position_embeddings is None:
        raise ValueError(
            "Transformers 4.51 must pass shared position_embeddings to attention"
        )
    cos, sin = position_embeddings
    query_states, key_states = _apply_rope(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    key_states, value_states = _crop_static_cache(
        key_states, value_states, past_key_value, cache_position, self.layer_idx
    )
    key_states = _repeat_kv(key_states, self.num_key_value_groups)
    value_states = _repeat_kv(value_states, self.num_key_value_groups)

    _capture_qk_if_requested(self, query_states, key_states)
    if q_len == key_states.shape[-2]:
        if batch != 1 and self.fastprefillconfig.metric != "full":
            raise ValueError("custom sparse prefill currently requires batch_size=1")
        attn_output = _run_prefill(
            self, query_states, key_states, value_states, attention_mask
        )
    else:
        # Decode remains exact; custom sparse kernels are only used for prefill.
        attn_output = _dense_attention(
            query_states, key_states, value_states, attention_mask
        )

    expected = (batch, num_heads, q_len, head_dim)
    if tuple(attn_output.shape) != expected:
        raise ValueError(
            f"attention output must have shape {expected}, got {tuple(attn_output.shape)}"
        )
    attn_output = self.o_proj(
        attn_output.transpose(1, 2).contiguous().reshape(batch, q_len, -1)
    )
    if started is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(hidden_states.device)
        print(
            f"[FastPrefill] layer={self.layer_idx} method={self.fastprefillconfig.metric} "
            f"q={q_len} k={key_states.shape[-2]} seconds={time.perf_counter()-started:.6f}",
            flush=True,
        )
    return attn_output, None


def load_model_451(
    *,
    name_or_path: str,
    fastprefillconfig: BaseFastPrefillConfig,
    expected_model_type: str,
    expected_layers: int,
    expected_heads: int,
    expected_kv_heads: int,
):
    model_config = AutoConfig.from_pretrained(
        name_or_path,
        trust_remote_code=True,
    )
    if fastprefillconfig.rope_scaling_type == "yarn":
        model_config.rope_scaling = {
            "rope_type": "yarn",
            "factor": fastprefillconfig.rope_factor,
            "original_max_position_embeddings": (
                fastprefillconfig.rope_original_max_position_embeddings
            ),
        }
        model_config.max_position_embeddings = int(
            fastprefillconfig.max_position_embeddings_override
            or math.ceil(
                fastprefillconfig.rope_original_max_position_embeddings
                * fastprefillconfig.rope_factor
            )
        )
        print(
            "[FastPrefill] rope_scaling=yarn "
            f"factor={fastprefillconfig.rope_factor} "
            "original_max_position_embeddings="
            f"{fastprefillconfig.rope_original_max_position_embeddings} "
            f"max_position_embeddings={model_config.max_position_embeddings}",
            flush=True,
        )
    elif fastprefillconfig.max_position_embeddings_override is not None:
        model_config.max_position_embeddings = int(
            fastprefillconfig.max_position_embeddings_override
        )

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        config=model_config,
        trust_remote_code=True,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    actual = (
        model.config.model_type,
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
    )
    expected = (
        expected_model_type,
        expected_layers,
        expected_heads,
        expected_kv_heads,
    )
    if actual != expected:
        raise ValueError(f"Expected model layout {expected}, got {actual}")
    if (
        expected_model_type == "qwen3"
        and fastprefillconfig.metric != "full"
        and getattr(model.config, "use_sliding_window", False)
        and getattr(model.config, "sliding_window", None) is not None
    ):
        raise ValueError("Sparse prefill does not implement Qwen3 sliding-window layers")

    for layer in model.model.layers:
        attention = layer.self_attn
        attention.fastprefillconfig = fastprefillconfig
        attention.forward = forward_eval_451.__get__(attention, type(attention))

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path, trust_remote_code=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
