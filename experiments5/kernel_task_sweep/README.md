# Multi-task kernel-size diagnostics at 128K

This is a new full-policy loss comparison, separate from existing experiments
3–5. Existing VT results are not modified. Activate `fyc_qwen` and launch from
the repository root. The backbone is Llama-3.1-8B-Instruct.

Six untrained kernels only:

```bash
bash experiments5/run_kernel_task_sweep_llama128k.sh --fixed-only --resume
```

Six fixed kernels plus a **matching Llama** learned checkpoint:

```bash
bash experiments5/run_kernel_task_sweep_llama128k.sh \
  --conv-weights /absolute/path/to/matching_llama_checkpoint.pt \
  --resume
```

Qwen step9750 contains 36 layers and cannot be used on the 32-layer Llama
backbone. The launcher checks checkpoint layout on CPU before data preparation.
No weights are cropped, averaged across layers, or silently substituted.
The existing Llama replay EMA is a compatible option if that is the intended
learned comparator; it is not relabeled as step9750.

Defaults:

- `qa_2`, `fwe`, `niah_single_1`, `niah_multikey_1` (QA2, FWE, NIAH-S1, NIAH-MK1).
- 2 samples per task, 8 independent prompts total; `--num-samples 1` is available
  for a quick 4-input check. Generation seed 20261008, target context 131072.
- Fixed 1x1, 3x3, 5x5, 7x7, 9x9, 11x11: center column union main diagonal,
  all active weights 1, shared across layers/heads, center counted once.
  Size k has 2k−1 active coefficients. Replicate padding matches deployment.
  1x1 preserves initial-score ranking.
- Top-k=0.65 and stride=8; each visible query row keeps ceil(0.65 * visible blocks),
  with future/padded keys removed and no sink/recent overrides. All policies
  have the same budget and independently select on their own activations.
- Every policy changes all layers/heads, not only the diagnostic head used in
  experiments 3/4. This is an untrained-size comparison plus an optional trained
  7x7 comparator, **not** six separately trained kernel variants.

Metrics:

1. Each fresh-cache evaluation measures mean teacher-forced answer-token NLL;
   prompt-only sparse prefill followed by ordinary answer decoding, no EOS loss.
2. QA2 uses the first stored reference (HotpotQA normally has one answer).
   FWE/NIAH serialize every required item in generator order with comma-space
   separators. No loss-driven choice of reference/order is performed.
3. Average NLL over inputs within each task, then average the four task means
   with equal weights. Also report macro NLL decrease from fixed1x1. Each policy
   uses exactly the same inputs/reference answers. This is not RULER accuracy.
   NLL depends on answer format; very small samples are descriptive diagnostics,
   not evidence of a general or statistically established advantage.
4. Every evaluation repeats NLL with its frozen prompt masks to record drift.
   Learned+fixed defaults yield 56 policy evaluations plus 56 repeat checks.

Data preparation calls existing RULER generators directly with explicit seed,
using the Llama template. QA2 requires existing HotpotQA source data; if missing,
follow the repository generator's message to run its data download script.
You can explicitly reuse existing Llama RULER data with `--raw-data-dir DIR`
(`DIR/TASK/validation.jsonl`), or converted records with `--data FILE`.
No templates are reapplied to already templated prompts. Prompt+answer token
counts are validated against the requested context; inputs are never truncated.
The resulting actual prompt token counts are recorded, as target 128K does not
mean every task has exactly 131072 prompt tokens.

Default results:

```
experiments5/kernel_task_results/llama128k_4tasks_2samples/
  run_manifest.json
  TASK/sN/fixed_1x1.json ... fixed_11x11.json [learned_checkpoint.json]
  individual_losses.csv
  task_mean_losses.csv
  summary.json
```

Repeat the same command with `--resume` after interruption. A completed
input-policy result is committed atomically after its repeat check; only
uncommitted evaluations are rerun. Settings/data/weight/model metadata and
source fingerprints must match. Changing from fixed-only to learned+fixed is
a different configuration and requires a new `--output-root`. Use distinct
output paths for distinct runs, and never run two processes against one path.
Data generation also reuses completed tasks with matching provenance.
GPU KV caches are not persisted; no new KV-offloading strategy is introduced.

Model loading runs in a temporary single-process environment: launcher rank
variables are hidden during `from_pretrained` and then restored, while
`CUDA_VISIBLE_DEVICES` and `device_map=auto` remain intact. This prevents
Transformers 4.51 from converting automatic model placement into tensor
parallelism when a scheduler supplies `WORLD_SIZE` without `LOCAL_RANK`.
Launch the sweep as one Python process, not multi-rank `torchrun`. The known
earlier loading-failure manifest can be upgraded only before any policy loss
has been saved; its original manifest is backed up. Prepared inputs are reused.

The default diagnostic block selector is now `--mask-backend cpu`: full padded
score estimation/refinement still runs on GPU, then real-block scores are
transferred to CPU for causal fixed-budget ranking. Ties favor smaller key-block
indices, consistently across every kernel. No CUDA Top-k/scatter is used for
selection. Attention and answer likelihood computation remain on GPU. This is
a loss diagnostic, not a throughput measurement; repository production
selectors and benchmark results are unchanged. `--mask-backend cuda` explicitly
uses the repository selector for debugging. CUDA synchronization after score
estimation and refinement identifies earlier asynchronous kernel failures.
The known failed sweep can migrate to this repair only before any loss result
is committed. Existing successful-policy results from another selector cannot
be mixed into the repaired run.

Output-path repair: the preparation script resolves generator output and
tokenizer paths to absolute paths before launching a generator in its own cwd.
An unfinished manifest from the known earlier path-bug version can be upgraded
only when every other parameter/source fingerprint matches; its original
manifest is backed up. Generated files misplaced beneath
`eval/RULER/scripts/data/experiments5/...` are copied into the correct data
directory and validated, preserving their originals. Completed data manifests
and conflicting task outputs are not silently replaced.

To check command/configuration without generating inputs or loading models:

```bash
bash experiments5/run_kernel_task_sweep_llama128k.sh --fixed-only --dry-run
```
# Nonfinite-score diagnostics

The sweep now distinguishes nonfinite fused block scores, nonfinite Q/K,
nonfinite convolution output, and nonfinite sparse-attention output. It writes
`nonfinite_diagnostic.json` in the run output directory. With finite Q/K only,
nonfinite fused scores are recomputed using an FP32 PyTorch reference for the
same inverse-antidiagonal sample sum, sampled-row causal softmax and block
aggregation. This is not dense attention and does not replace NaNs with zeros.
Workspace is limited to one head and 128 sampled query rows. Reference recovery
can be substantially slower; recovered layers are recorded in each policy JSON.
If Q/K or sparse-attention output is nonfinite, the sweep stops with the layer
and stage instead of saving a loss. Production RULER/LongBench code is unchanged.

An exact known failed prior version may resume only if no policy losses have
been saved. Runs with completed results require a new output directory when the
source changes. Run `python experiments5/kernel_task_sweep/test_sweep.py` in
`fyc_qwen` to execute the additional CPU PyTorch numerical-oracle test; that
test is skipped when PyTorch is unavailable. GPU validation is still required.

