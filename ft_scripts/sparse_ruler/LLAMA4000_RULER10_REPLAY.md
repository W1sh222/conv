# Llama4000 continuation: RULER 128K with LongBench retention

Run from the repository root in the server's `fyc_qwen` environment:

```bash
bash ft_scripts/sparse_ruler/run_llama4000_ruler10_replay.sh
```

The default input is the **evaluated continuation EMA**:
`xattn/conv_weights/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_96k128k_continue_from4000_v1_bf16_ema.pt`.
This is distinct from the older refinement checkpoint ending in `_ema_step4000.pt`.
The original launcher and input weights remain unchanged. New outputs go to:
`xattn/conv_weights/llama4000_ruler10_replay_topp092_v1/`.

## Training choices

- Frozen Llama-3.1-8B backbone; only the existing 32-layer/32-head 7x7 kernels change.
- Native Llama RoPE, block size 128, score stride 8, BF16, one sampled layer per update.
- Main stream: fresh synthetic 112K–128K prompts for the ten scored RULER tasks.
  The excluded tasks are `niah_multikey_2`, `niah_multikey_3`, and `qa_1`.
- Main mix: singles 9%, multikey_1 15%, multivalue 10%, multiquery 10%,
  vt 15%, cwe 20%, fwe 10%, qa_2 11%. This reallocates the old 42% spent on excluded tasks.
  CWE now asks for ten distinct common words, matching the RULER answer count,
  instead of the original three-word surrogate. Synthetic text and distractor
  construction still differ from the official benchmark.
- Replay: exactly one of every four successful optimizer updates uses an independent
  8K–64K stream. QA, aggregation, retrieval and general prompts are retained.
  The default replay is a **synthetic surrogate**, not actual LongBench training data.
  You can supply independent domain-matched training JSONL via `--replay-data PATH`.
  JSONL records use `messages` or `text`; optional `meta` supplies evidence spans.
  Do not use LongBench or RULER evaluation examples as training replay.
- Evaluation-matched hard-mask diagnostics use corrected `positive_mass_v1`
  Top-p=0.92 with no fixed Top-k override. Dense-teacher KL also includes a positive
  score-distribution term. Existing Top-k boundary loss is a small auxiliary loss,
  **not** the selector used to report training recall/density.
- Replay distillation anchors both the softplus and positive score distributions to
  the fixed input kernel. Kernel drift penalty and the elementwise residual bound
  (`alpha=0.02`) limit changes. These reduce forgetting risk; they do not guarantee
  a LongBench score or a RULER improvement.
- Default: 2000 total updates (1500 main, 500 replay), LR 2e-7, cosine decay,
  200 warmup updates, EMA 0.999, checkpoint every 250 updates.

The objective remains block-level distillation plus synthetic evidence coverage;
it is **not** end-to-end answer cross-entropy optimization. Final task scores must
be measured after training.

## Preview, short smoke run, resume

```bash
bash ft_scripts/sparse_ruler/run_llama4000_ruler10_replay.sh --dry-run
# Use a separate directory so smoke outputs do not block the full run:
bash ft_scripts/sparse_ruler/run_llama4000_ruler10_replay.sh \
  --steps 4 --run-dir xattn/conv_weights/llama4000_ruler10_replay_smoke

bash ft_scripts/sparse_ruler/run_llama4000_ruler10_replay.sh \
  --resume-state xattn/conv_weights/llama4000_ruler10_replay_topp092_v1/conv_kernel_7x7_llama4000_ruler10_replay_train_state.pt
```

Resume preserves the optimizer, fixed anchor, EMA, Python RNG, sample permutation,
sample pointers and layer queue. It rejects changes to training configuration.
Do not pass the original continuation's optimizer state to this new experiment.
The smoke run still builds the default-size datasets; override `MAIN_SAMPLES` and
`REPLAY_SAMPLES` for a quicker data-generation smoke run.

Check local or server installation before running numerical tests:

```bash
python -m unittest discover -s ft_scripts/sparse_ruler -p test_replay_continuation.py -v
```

## Evaluation and checkpoint choice

Evaluate the input anchor and saved EMA candidates at the **same Top-p=0.92**.
Use independent validation samples for choosing a checkpoint; run final benchmark
evaluation after selecting it. Do not assume the last EMA is best.

```bash
WEIGHT=xattn/conv_weights/llama4000_ruler10_replay_topp092_v1/conv_kernel_7x7_llama4000_ruler10_replay_ema.pt
bash scripts/run_ruler_conv2_9_parallel.sh --method conv --weight "$WEIGHT" --topp 0.92 --stride 8
bash scripts/run_longbench_conv.sh --conv_weight_path "$WEIGHT" --top_p 0.92 --stride 8
```

These existing evaluators run their usual full suites; compute the RULER ten-task
mean afterwards. The checkpoint selector requires a complete 16-task LongBench
JSON and a complete 13-task RULER CSV, and applies the exclusions itself.
It retains null-output scores and does not fabricate missing results.

Create `candidates.json` with validation score paths:

```json
[
  {
    "weight": "/path/to/candidate_ema_step1000.pt",
    "longbench": "/path/to/candidate_longbench/result.json",
    "ruler128k": "/path/to/candidate_ruler128k/summary.csv",
    "top_p": 0.92,
    "selector": "positive_mass_v1"
  }
]
```

```bash
python ft_scripts/sparse_ruler/select_replay_checkpoint.py \
  --baseline-longbench /path/to/anchor_longbench/result.json \
  --baseline-ruler128k /path/to/anchor_ruler128k/summary.csv \
  --candidates candidates.json --topp 0.92 \
  --max-longbench-drop 0 --min-ruler-gain 0.1 \
  --out xattn/conv_weights/llama4000_ruler10_replay_topp092_v1/selection.json
```

Default acceptance: no LongBench **mean** regression and at least 0.1 point of
RULER ten-task gain. The best eligible RULER candidate is recommended; if none
qualifies, `keep_input_anchor` is returned. No weight file is overwritten.
For reference, the currently recorded anchor benchmark means are LongBench 37.703125
and RULER128K ten-task 77.45. Validation means can differ; pass their actual files.
