# RULER Top-p audit (2026-10-03)

Audited local counterpart of the supplied command:
`scripts/run_ruler_conv2_9_parallel.sh --weight ...continue_from4000...pt --topp 0.95`.

## Confirmed control flow

The parallel launcher exports `RULER_SPARSE_TOP_P` and passes `--threshold 0.95` to each shard. `call_api.py` gives an explicit threshold priority over the default block ratio, setting `block_topk_ratio=None`. Generation uses a separate `--top_p=1.0` and greedy decoding. The explicitly supplied Conv path is passed through the Llama adapter; a missing explicit checkpoint raises an error instead of silently loading an initial kernel. Decode uses dense attention. No selector override or generation top-p mix-up was found in this path.

## Two selector defects reproduced

`Conv.py` refines the block map and passes the raw signed result to `utils.py:find_blocks_chunked`. The latter computes the required mass from the entire unmasked row. This requires nonnegative, already causal scores, conditions not guaranteed after convolution.

1. Negative entries cancel positive mass. For `[.1,.5,.25,.15,.05,-.3,-.25,.1]`, at p=.95 the existing selector retains indices 0,1,7 and only 60.87% of causal positive mass. This is not a 95% probability guarantee. It remains a concrete input-domain defect even though the actual checkpoint's negative-score frequency has not yet been measured.
2. Future entries contribute to total mass and ranking before the final causal cleanup. For a row with four visible blocks and scores `[.1,.2,.3,.4,20,0,0,0]`, the future block is selected and then removed; only indices 0 and 3 remain, covering 50% of the valid mass. Token attention remains causal; the defect is lost selection coverage, not token-level leakage. Actual future-score magnitude matters. A nonnegative uniform causal-map example with the real vertical/diagonal initialization creates future entries but does not fail the coverage criterion; future nonzeros alone are not proof of an accuracy impact.

The unmodified function was extracted from `utils.py` and executed using a NumPy API adapter locally because neither local environment has PyTorch. The nonnegative causal control passes. The same script runs with real PyTorch on the evaluation host. These tests establish selector counterexamples, not the cause or magnitude of a complete RULER run.

Training's teacher-distribution loss first applies softplus with temperature .015 and masks future keys before normalization. Inference Top-p skips that transformation. Top-k depends on score order and a fixed count, so it does not rely on signed sums. This helps explain why good Top-k results do not certify the Top-p implementation or its calibration.

## Existing result evidence

Unweighted averages of the thirteen local task scores:

| Length | Conv Top-k=.70 | Conv Top-k=.75 | Conv Top-p=.95 |
|---|---:|---:|---:|
|32K|75.62|82.87|88.53|
|64K|83.67|86.10|81.26|
|128K|79.55|81.18|73.55|

Thus the drop is context- and task-dependent. CWE at 32K/64K/128K falls from 80.5/30.2/8.2 (Top-k=.70) to 54.0/2.9/0.7 (Top-p=.95). At 128K, the three single-needle tasks remain 100/100/96. The broad-coverage and multi-key tasks warrant closer inspection. These task differences are not proof that selector defects alone explain the loss.

Top-p=.95 means cumulative score mass, not retaining 95% of blocks or 95% of useful context. Even a correct probability selector can keep few blocks on a peaked distribution. The checkpoint was trained at a fixed ratio; probability calibration and downstream utility are separate questions.

## Run the selector and checkpoint checks

On the server, after activating its existing environment, from the repository root:

```bash
python experiments/audit_ruler_topp.py \
  --weight xattn/conv_weights/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_96k128k_continue_from4000_v1_bf16_ema.pt \
  --topp 0.95
```

## Probe one actual prompt without modifying selection

Use a fresh prediction directory. This wrapper only adds score/mask diagnostics and processes the first sample; it preserves the production selector and generation. It incurs extra diagnostic overhead and must not be used for timing benchmarks.

Version `2026-10-03.3` crops statistics to the actual query/key block counts and uses the same mask cleanup as inference. Older probe versions included chunk-padding blocks in the denominator and treated zero positive mass as 100% coverage. Consequently, their `last_row_*` values can describe a padded row, and their reported density/coverage should not be used as actual-input metrics. Reprobe the same first sample after updating the script; no full benchmark rerun is needed for this diagnostic correction.

```bash
python experiments/probe_ruler_topp.py \
  --audit-output output/ruler_topp_audit/cwe64k/actual_layers.jsonl \
  --dump-layer 16 --max-samples 1 -- \
  --data_dir eval/RULER/scripts/data/ruler_926/synthetic/65536/data \
  --save_dir output/ruler_topp_audit/cwe64k/pred \
  --benchmark synthetic --task cwe --server_type hf \
  --model_name_or_path /inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct \
  --temperature 0 --top_k 32 --top_p 1 --batch_size 1 \
  --metric conv --stride 8 --threshold 0.95 \
  --conv_weight_path xattn/conv_weights/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_96k128k_continue_from4000_v1_bf16_ema.pt \
  --attention_implementation sdpa
```

Inspect `query_tokens`, `real_query_blocks`, `padding_query_blocks`, `causal_density`, `negative_valid_fraction`, `future_abs_score_fraction`, `padding_key_abs_score_fraction`, positive/softplus coverage, and last-query-row counts per layer. Coverage is a diagnostic of transformed Conv scores, not measured full-attention recall or answer accuracy. Positive-mass coverage excludes zero-mass rows and reports their fraction separately; an entirely undefined summary is JSON `null`. Softplus is a comparison using the specified temperature, not proof that this particular checkpoint used that training configuration.

The dump preserves the full score tensor, adds `layer16_refined_scores.metadata.json` with actual dimensions, and saves `layer16_actual_mask.pt` after inference mask cleanup. The saved score map can also be checked independently:

```bash
python experiments/audit_ruler_topp.py \
  --scores output/ruler_topp_audit/cwe64k/layer16_refined_scores.pt \
  --offset 0 --topp 0.95 \
  --output output/ruler_topp_audit/cwe64k/selector_report.json
```

The standalone audit runs the selector in a single query chunk. Production selects separately for each `chunk_size/block_size` query block group and then crops/cleans the mask. Its full-map result therefore is not an exact reproduction of production masks. Use the probe's actual-mask diagnostics for that purpose.

The production scripts resume old predictions by index and do not validate checkpoint hashes. A changed checkpoint at the same path, or a changed selector under an old output tag, can reuse stale outputs. Use a new run tag for any rerun or implementation change; do not delete previous experiment records.

## Implemented correction (2026-10-03)

`xattn/src/conv_topp.py` defines the corrected Conv Top-p selector. It crops to real query/key blocks before selection, masks future positions, sets negative scores to zero, includes forced sink/diagonal mass exactly once, and includes the first threshold-crossing candidate. Zero-positive-mass rows retain all causally valid blocks; p=1 is a dense causal control. This guarantees coverage of the defined positive Conv mass (up to floating-point tolerance), not full-attention recall or improved answer accuracy. No softplus temperature was introduced: softplus remains a separate calibration choice to evaluate against the checkpoint's actual training configuration.

`Conv.py` uses this policy by default only when threshold selection is active. Its ratio Top-k, fixed-count Top-k, safe-Top-k, and XAttention's shared `find_blocks_chunked` code are unchanged. The training-side `make_inference_chunked_block_mask` uses the same corrected Top-p helper; fixed-ratio training remains unchanged. Existing checkpoints, benchmark scores, figures, and user training-script edits were not rewritten.

Both RULER parallel launchers and `scripts/run_longbench_451.sh` append `positive_mass_v1` to corrected Conv Top-p experiment tags (including custom tags), keeping prior predictions separate. LongBench's final result-tag argument takes precedence over earlier forwarded copies. Direct calls to `call_api.py`, `pred.py`, lower-level launchers, and efficiency scripts still require a fresh output location chosen by the user.

The probe version `2026-10-03.4` adds `conv_topp_selector` to each record; the real-block statistics introduced in .3 remain the same. Synchronize **both** `xattn/src/Conv.py` and the new `xattn/src/conv_topp.py` before probing; synchronize the changed launchers before benchmarking.

Default corrected run:

```bash
export CONV_TOPP_SELECTOR=positive
python experiments/test_conv_topp_selector.py --invariants-only
bash scripts/run_ruler_conv2_9_parallel.sh \
  --weight xattn/conv_weights/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_96k128k_continue_from4000_v1_bf16_ema.pt \
  --topp 0.95
```

For an old-selector comparison, retain the original logic but write to a fresh custom tag:

```bash
CONV_TOPP_SELECTOR=legacy bash scripts/run_ruler_conv2_9_parallel.sh \
  --weight xattn/conv_weights/conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_96k128k_continue_from4000_v1_bf16_ema.pt \
  --topp 0.95 --run-tag cwe_legacy_comparison_new
```

Selector regression tests passed locally through a NumPy API adapter, including causal mass coverage, chunk padding, forced/crossing blocks, zero-mass fallback, per-head thresholds, monotonic selection, unchanged Top-k masks, legacy reproduction, and training/inference alignment. Bash syntax checks passed. The local environment lacks PyTorch/CUDA and the model/checkpoint; GPU execution, new LongBench/RULER scores, and new speedups are **not verified**. Run the same single-sample probe in a fresh prediction directory first, then measure task results and speed separately.
