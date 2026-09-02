#!/usr/bin/env python3
"""Validate a Qwen3-8B 36-layer Conv checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "sparse_ruler"
sys.path.insert(0, str(COMMON_DIR))

from verify_conv_kernel import main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    main(default_num_layers=36)
