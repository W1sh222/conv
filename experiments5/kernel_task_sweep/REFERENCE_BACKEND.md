# Reference numerical protocol v2

The previous remaining-nine-task run had zero paired inputs, illegal CUDA memory
accesses, and impossible nonfinite-element counts. The exact custom-kernel root
cause is not established. Increasing the number of samples alone cannot repair it.

The diagnostic runner now defaults to `--score-backend reference` and
`--attention-backend sdpa`. Sampled scoring uses the existing inverse-antidiagonal
FP32 reference with causal normalization and block aggregation. Attention uses
the selected block mask, exact token-level causality, and PyTorch SDPA, processing
128 queries and one head at a time. It never materializes an N-by-N token mask.
Convolution, budget, labels and paired aggregation remain unchanged.

This trades speed for a reference computation and is not an efficiency benchmark.
The server must validate CUDA execution. CPU tests compare reference attention
against explicit dense masked softmax, including a partial final block and head
chunking. Run before a long job:

```bash
python -m unittest discover -s experiments5/kernel_task_sweep -p 'test_safe_attention.py'
```

First retry one failed task/input in a new directory using its existing data:

```bash
bash experiments5/run_kernel_task_sweep_llama128k.sh \
  --fixed-only --tasks niah_single_2 --num-samples 1 \
  --data experiments5/kernel_task_data/llama_131072_remaining9tasks_2samples_seed20261008/observation.jsonl \
  --output-root experiments5/kernel_task_results/llama128k_reference_v2_smoke \
  --kernel-sizes 1 --isolate-samples --resume
```

After the smoke test, use all 13 tasks with four inputs/task:

```bash
bash experiments5/run_kernel_task_sweep_ruler_all.sh --all-tasks --resume
```

Every jointly successful input is retained; a failed input is excluded from all
policies. Four is an increased predeclared input count, not a guarantee of four
successful pairs. Inspect `samples_per_task`, `paired_samples`, `failures`, and
`usable_for_kernel_comparison`. Do not silently add samples until a favorable
kernel ranking appears. If failures remain, diagnose them before increasing the
count again. No NaNs are replaced with zeros.

Old directories are preserved. Run manifests prevent mixing numerical protocols.
Do not merge the old four-task custom-backend NLLs with these new reference NLLs;
rerun those four tasks using the same reference protocol (the all-task command
does that). Official RULER/LongBench benchmark selectors and results are untouched.
