# Qwen3 origin-kernel LongBench refinement

This branch starts from the deterministic original vertical-plus-main-diagonal
kernel. It first trains on an 8K-64K native-RoPE LongBench surrogate, then
runs a small 96K-128K YaRN RULER replay. Both phases keep block size 128,
score stride 8, and top-k ratio 0.65. Phase-1 EMA is the LongBench candidate;
phase-2 EMA is the RULER-protected candidate.
