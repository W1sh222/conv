#!/usr/bin/env python3
"""Write the deterministic original vertical-plus-diagonal Qwen Conv kernel."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

def make_origin(layers: int, heads: int, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    base = torch.zeros(kernel_size, kernel_size, dtype=torch.float32)
    center = kernel_size // 2
    base[:, center] = 1.0
    idx = torch.arange(kernel_size)
    base[idx, idx] = 1.0
    return base[None, None].repeat(layers, heads, 1, 1).contiguous()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=7)
    a = p.parse_args()
    w = make_origin(a.layers, a.heads, a.kernel_size)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(w, out)
    print(f"[origin] wrote {out} shape={tuple(w.shape)} nonzero={int(torch.count_nonzero(w))}")

if __name__ == "__main__":
    main()
