#!/usr/bin/env python3
"""Qwen3 tokenizer entry point for the shared RULER-Mix data builder."""

from __future__ import annotations

import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "sparse_ruler"
sys.path.insert(0, str(COMMON_DIR))

from build_ruler_mix_sft import main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    main()
