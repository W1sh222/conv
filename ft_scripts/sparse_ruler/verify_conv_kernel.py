#!/usr/bin/env python3
"""Validate an inference-ready per-layer/per-head Conv checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def verify(path: str, num_layers: int, num_heads: int, kernel_size: int) -> None:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    weight = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = (num_layers, num_heads, kernel_size, kernel_size)
    if not torch.is_tensor(weight):
        raise TypeError(f"checkpoint must contain a tensor, got {type(weight)}")
    if tuple(weight.shape) != expected:
        raise ValueError(f"expected shape {expected}, got {tuple(weight.shape)}")
    if not torch.isfinite(weight).all():
        raise ValueError("checkpoint contains NaN or Inf")
    weight = weight.float()
    print(
        f"[verify] path={checkpoint} shape={tuple(weight.shape)} "
        f"min={weight.min().item():.6f} max={weight.max().item():.6f} "
        f"mean={weight.mean().item():.6f} std={weight.std().item():.6f}"
    )


def main(default_num_layers: int = 32) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--num_layers", type=int, default=default_num_layers)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=7)
    args = parser.parse_args()
    verify(args.path, args.num_layers, args.num_heads, args.kernel_size)


if __name__ == "__main__":
    main()
