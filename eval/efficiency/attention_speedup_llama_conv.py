"""Upstream-style Llama top-p speed test with an additional Conv column.

This is intentionally a new entry point.  It does not modify the upstream
``attention_speedup.py`` or any existing inference loader.  Q/K are captured
once per length, then Full, Flex, XAttention and Conv are timed on identical
Q/K/V tensors.  Llama only; Conv uses the same 128-token block and threshold
(top-p) selection contract as XAttention.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import torch

from xattn.efficiency_methods.llama_methods import (
    DEFAULT_CONV_WEIGHT,
    DEFAULT_FLEX_GAMMA,
    DEFAULT_FLEX_TAU,
    DEFAULT_THRESHOLD,
    benchmark_prefill,
    estimate_density,
)
from xattn.efficiency_methods.llama_qk_cache import capture_qk


DEFAULT_MODEL = "/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("LLAMA_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--lengths", default=os.environ.get("EFFICIENCY_LENGTHS", "4,8,16,32,64,128"))
    parser.add_argument("--cache-dir", default=os.environ.get("LLAMA_QK_CACHE_DIR", "output/efficiency_llama_conv"))
    parser.add_argument("--layer", type=int, default=int(os.environ.get("EFFICIENCY_LAYER", "12")))
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("EFFICIENCY_ITERATIONS", "50")))
    parser.add_argument("--warmups", type=int, default=int(os.environ.get("EFFICIENCY_WARMUPS", "30")))
    parser.add_argument("--capture-chunk-tokens", type=int, default=int(os.environ.get("CAPTURE_CHUNK_TOKENS", "2048")))
    parser.add_argument("--method-chunk-size", type=int, default=int(os.environ.get("METHOD_CHUNK_SIZE", "32768")))
    parser.add_argument("--stride", type=int, choices=(8, 16), default=int(os.environ.get("STRIDE", "8")))
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("TOPP_THRESHOLD", str(DEFAULT_THRESHOLD))))
    parser.add_argument("--conv-weight-path", default=os.environ.get("CONV_WEIGHT_PATH", DEFAULT_CONV_WEIGHT))
    parser.add_argument("--flex-gamma", type=float, default=float(os.environ.get("FLEX_GAMMA", str(DEFAULT_FLEX_GAMMA))))
    parser.add_argument("--flex-tau", type=float, default=float(os.environ.get("FLEX_TAU", str(DEFAULT_FLEX_TAU))) )
    parser.add_argument("--minference-vertical-size", type=int, default=int(os.environ.get("MINFERENCE_VERTICAL_SIZE", "1000")))
    parser.add_argument("--minference-slash-size", type=int, default=int(os.environ.get("MINFERENCE_SLASH_SIZE", "6096")))
    parser.add_argument("--full-backend", choices=("flashinfer", "sdpa"), default=os.environ.get("FULL_BACKEND", "flashinfer"))
    parser.add_argument("--offload-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rope-scaling-type", choices=("none", "yarn"), default=os.environ.get("ROPE_SCALING_TYPE", "none"))
    parser.add_argument("--rope-factor", type=float, default=float(os.environ.get("ROPE_FACTOR", "4.0")))
    parser.add_argument("--rope-original-max-position-embeddings", type=int, default=int(os.environ.get("ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS", "32768")))
    parser.add_argument("--max-position-embeddings", type=int, default=int(os.environ.get("MAX_POSITION_EMBEDDINGS", "131072")))
    parser.add_argument("--result-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.iterations <= 0 or args.warmups < 0:
        raise ValueError("iterations must be positive and warmups non-negative")
    if args.capture_chunk_tokens <= 0 or args.method_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if not 0.0 < args.flex_gamma <= 1.0 or args.flex_tau < 0.0:
        raise ValueError("Flex gamma must be in (0,1] and tau non-negative")
    if args.minference_vertical_size <= 0 or args.minference_slash_size <= 0:
        raise ValueError("MInference vertical/slash sizes must be positive")

    lengths = [int(x.strip()) * 1024 for x in args.lengths.split(",") if x.strip()]
    if not lengths or any(x <= 0 for x in lengths):
        raise ValueError("--lengths must contain positive K values")
    cache_dir = Path(args.cache_dir)
    # Keep the same method set as the upstream efficiency benchmark and add
    # Conv.  Minference is included explicitly so the summary table below is
    # complete (and does not depend on a missing dictionary key).
    methods = ("full", "flex", "xattn", "conv", "minference")
    print(
        f"[Llama efficiency] model={args.model_path} lengths={[x // 1024 for x in lengths]}K "
        f"stride={args.stride} threshold={args.threshold} "
        f"conv_weight={args.conv_weight_path} full={args.full_backend} "
        f"flex_gamma={args.flex_gamma} flex_tau={args.flex_tau} "
        f"offload_cache={args.offload_cache}",
        flush=True,
    )

    rows = []
    for target_len in lengths:
        q_cpu, k_cpu = capture_qk(
            model_path=args.model_path,
            cache_dir=cache_dir,
            target_len=target_len,
            layer=args.layer,
            chunk_tokens=args.capture_chunk_tokens,
            offload_cache=args.offload_cache,
            rope_scaling_type=args.rope_scaling_type,
            rope_factor=args.rope_factor,
            rope_original_max_position_embeddings=args.rope_original_max_position_embeddings,
            max_position_embeddings=args.max_position_embeddings,
        )
        q = q_cpu.to("cuda", dtype=torch.bfloat16, non_blocking=True).contiguous()
        k = k_cpu.to("cuda", dtype=torch.bfloat16, non_blocking=True).contiguous()
        del q_cpu, k_cpu
        torch.manual_seed(0)
        v = torch.randn_like(q).contiguous()
        torch.cuda.synchronize()

        times = {}
        densities = {}
        for method in methods:
            try:
                times[method] = benchmark_prefill(
                    method,
                    q,
                    k,
                    v,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    stride=args.stride,
                    threshold=args.threshold,
                    chunk_size=min(args.method_chunk_size, target_len),
                    conv_weight_path=args.conv_weight_path,
                    conv_layer=args.layer,
                    flex_gamma=args.flex_gamma,
                    flex_tau=args.flex_tau,
                    minference_vertical_size=args.minference_vertical_size,
                    minference_slash_size=args.minference_slash_size,
                    full_backend=args.full_backend,
                )
                if method in {"xattn", "conv"}:
                    densities[method] = estimate_density(
                        method,
                        q,
                        k,
                        v,
                        stride=args.stride,
                        threshold=args.threshold,
                        chunk_size=min(args.method_chunk_size, target_len),
                        conv_weight_path=args.conv_weight_path,
                        conv_layer=args.layer,
                        flex_gamma=args.flex_gamma,
                        flex_tau=args.flex_tau,
                    )
            except Exception as exc:
                print(f"[WARN] {method} failed at {target_len // 1024}K: {exc!r}", flush=True)
                times[method] = float("nan")
                densities[method] = None

        full_time = times["full"]
        speedups = {
            method: (full_time / value if math.isfinite(full_time) and math.isfinite(value) and value > 0 else float("nan"))
            for method, value in times.items()
        }
        row = {
            "length_tokens": target_len,
            "length_k": target_len // 1024,
            "times_sec": times,
            "speedup_vs_full": speedups,
            "density": densities,
        }
        rows.append(row)
        print(
            f"{target_len // 1024}K "
            + " ".join(f"{m}={times[m]:.4f}s" for m in methods)
            + " | "
            + " ".join(f"{m}_x={speedups[m]:.2f}" for m in methods if m != "full"),
            flush=True,
        )
        del q, k, v
        gc.collect()
        torch.cuda.empty_cache()

    print("\nLength    Flex    Xattn   Conv    Minfer  Full")
    for row in rows:
        s = row["speedup_vs_full"]
        print(
            f"{row['length_k']:>5}K    {s['flex']:>5.2f}   {s['xattn']:>5.2f}   "
            f"{s['conv']:>5.2f}   {s['minference']:>5.2f}   1.00"
        )
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[saved] {result_path}")


if __name__ == "__main__":
    main()
