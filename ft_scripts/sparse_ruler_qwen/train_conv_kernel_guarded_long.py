#!/usr/bin/env python3
"""Qwen3-8B entry point for the shared Transformers 4.51 trainer."""

from __future__ import annotations

import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "sparse_ruler"
sys.path.insert(0, str(COMMON_DIR))

from train_conv_kernel_guarded_long import main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    main(default_model_type="qwen3")
