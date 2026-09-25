"""Q/K capture with bounded GPU memory for the standalone Llama benchmark."""

from __future__ import annotations

import gc
import inspect
import pickle
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache, StaticCache

from eval.efficiency.generate_prompt import generate_prompt
from xattn.src.load_llama import FastPrefillConfig, load_model


def _make_static_cache(model, max_cache_len: int):
    """Support both Transformers 4.46 and 4.51 StaticCache signatures."""
    params = inspect.signature(StaticCache).parameters
    kwargs = {
        "config": model.config,
        "max_cache_len": int(max_cache_len),
        # With device_map="balanced" the first parameter can be on CPU while
        # the embedding/input device is CUDA.  StaticCache must follow the
        # input device or the first cached chunk will trigger a device copy.
        "device": model.model.embed_tokens.weight.device,
        "dtype": model.dtype,
    }
    if "max_batch_size" in params:
        kwargs["max_batch_size"] = 1
    else:
        kwargs["batch_size"] = 1
    return StaticCache(**kwargs)


def _make_cache(model, target_len: int, offload: bool):
    if offload:
        # DynamicCache offloading keeps inactive layer KV tensors on CPU.  It
        # is slower than StaticCache but prevents a 128K capture from reserving
        # a full GPU KV cache for every layer.
        try:
            params = inspect.signature(DynamicCache).parameters
            kwargs = {}
            if "config" in params:
                kwargs["config"] = model.config
            if "offloading" in params:
                kwargs["offloading"] = True
            else:
                raise TypeError("this Transformers version has no DynamicCache offloading")
            return DynamicCache(**kwargs)
        except (TypeError, ValueError):
            print("[capture] DynamicCache(offloading=True) unavailable; using StaticCache", flush=True)
    return _make_static_cache(model, target_len)


def _cache_paths(cache_dir: Path, target_len: int) -> Tuple[Path, Path]:
    return cache_dir / f"query_{target_len}.pkl", cache_dir / f"key_{target_len}.pkl"


def load_cached_qk(cache_dir: Path, target_len: int):
    query_path, key_path = _cache_paths(cache_dir, target_len)
    if not query_path.exists() or not key_path.exists():
        return None
    with query_path.open("rb") as handle:
        q = pickle.load(handle)
    with key_path.open("rb") as handle:
        k = pickle.load(handle)
    if tuple(q.shape[-2:]) != (target_len, q.shape[-1]):
        raise ValueError(f"cached query has wrong length: {tuple(q.shape)}")
    if tuple(k.shape[-2:]) != (target_len, k.shape[-1]):
        raise ValueError(f"cached key has wrong length: {tuple(k.shape)}")
    return q, k


@torch.inference_mode()
def capture_qk(
    *,
    model_path: str,
    cache_dir: Path,
    target_len: int,
    layer: int,
    chunk_tokens: int,
    offload_cache: bool,
    rope_scaling_type: str,
    rope_factor: float,
    rope_original_max_position_embeddings: int,
    max_position_embeddings: Optional[int],
):
    """Capture one layer's post-RoPE Q/K while chunking the model forward."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = load_cached_qk(cache_dir, target_len)
    if cached is not None:
        q, k = cached
        return q.contiguous(), k.contiguous()

    config = FastPrefillConfig(
        metric="full",
        stride=8,
        attention_implementation="sdpa",
        rope_scaling_type=rope_scaling_type,
        rope_factor=rope_factor,
        rope_original_max_position_embeddings=rope_original_max_position_embeddings,
        max_position_embeddings_override=max_position_embeddings,
    )
    model, tokenizer = load_model(config, name_or_path=model_path)
    model.eval()
    for attention in (layer_.self_attn for layer_ in model.model.layers):
        attention.layer_to_save = int(layer)
        attention.target_len = int(target_len)
        attention.capture_output_dir = str(cache_dir)

    input_ids = generate_prompt(tokenizer, target_len)
    if input_ids.shape[1] != target_len:
        raise ValueError(
            f"prompt generator returned {input_ids.shape[1]} tokens, expected {target_len}"
        )
    input_device = model.model.embed_tokens.weight.device
    past_key_values = _make_cache(model, target_len, offload_cache)

    print(
        f"[capture] length={target_len} chunk={chunk_tokens} "
        f"cache={'offloaded-dynamic' if offload_cache else 'static'}",
        flush=True,
    )
    for start in tqdm(range(0, target_len, chunk_tokens), desc=f"capture {target_len}", unit="chunk"):
        stop = min(start + chunk_tokens, target_len)
        chunk = input_ids[:, start:stop].to(input_device, non_blocking=True)
        cache_position = torch.arange(start, stop, device=input_device, dtype=torch.long)
        model(
            input_ids=chunk,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            num_logits_to_keep=1,
        )
        del chunk, cache_position
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    query_path, key_path = _cache_paths(cache_dir, target_len)
    with query_path.open("rb") as handle:
        q = pickle.load(handle)
    with key_path.open("rb") as handle:
        k = pickle.load(handle)
    if q.shape[-2] != target_len or k.shape[-2] != target_len:
        raise RuntimeError(f"capture files have wrong shapes: q={q.shape}, k={k.shape}")

    del model, tokenizer, input_ids, past_key_values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return q.contiguous(), k.contiguous()
