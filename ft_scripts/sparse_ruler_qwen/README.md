# Qwen3-8B sparse RULER Conv training

This folder contains Qwen3-specific entry points. The optimization objective,
block scorer, loss functions, and checkpoint format are shared with
`../sparse_ruler` so Llama and Qwen3 cannot silently drift apart.

The shared Transformers 4.51 adapter applies Qwen3 `q_norm`/`k_norm` before
RoPE and validates the expected `(36 layers, 32 Q heads, 8 KV heads)` layout.
The output tensor shape is `[36, 32, 7, 7]`.
