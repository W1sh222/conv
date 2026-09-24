# Qwen3-8B sparse RULER Conv training

This folder contains Qwen3-specific entry points. The optimization objective,
block scorer, loss functions, and checkpoint format are shared with
`../sparse_ruler` so Llama and Qwen3 cannot silently drift apart.

The shared Transformers 4.51 adapter applies Qwen3 `q_norm`/`k_norm` before
RoPE and validates the expected `(36 layers, 32 Q heads, 8 KV heads)` layout.
The output tensor shape is `[36, 32, 7, 7]`.

`../run_ruler_mix_sparse_guarded_qwen.sh` is the balanced 64K-128K
continuation pipeline. It starts from the previous 48K-64K step16000 EMA,
uses static YaRN factor 4 in every stage, and emits separate 64K, 96K, 128K,
and mixed-length checkpoints. The matching evaluation entry point is
`../../scripts/run_ruler_qwen3_64k128k.sh`.
Use `../../scripts/run_longbench_qwen3_balanced.sh` for the matching LongBench
checkpoint; it keeps native RoPE by default and exposes an opt-in YaRN mode for
datasets whose prompts actually exceed the native context.

For the current LongBench-focused continuation, use
`../run_ruler_mix_sparse_guarded_qwen_stage4.sh`.  It always requires the
Stage-3 EMA ending in `step9250.pt`, writes a new
`conv_qwen3_t065_longbench_stage4_v2` directory, and leaves the Stage-3 RULER
checkpoint and the old `compensate_stage4_v1` branch untouched.  The companion
`../run_ruler_mix_sparse_guarded_qwen_combined.sh` first makes/validates Stage 3
and then invokes this isolated Stage-4 branch.
