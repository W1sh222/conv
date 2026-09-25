"""Standalone Llama efficiency-benchmark helpers.

This package intentionally lives beside :mod:`xattn.src` so the original
upstream files and the repository's inference paths remain untouched.
"""

from .llama_methods import (
    DEFAULT_CONV_WEIGHT,
    DEFAULT_FLEX_GAMMA,
    DEFAULT_FLEX_TAU,
    DEFAULT_THRESHOLD,
    benchmark_prefill,
    call_method,
)

__all__ = [
    "DEFAULT_CONV_WEIGHT",
    "DEFAULT_FLEX_GAMMA",
    "DEFAULT_FLEX_TAU",
    "DEFAULT_THRESHOLD",
    "benchmark_prefill",
    "call_method",
]
