# Exhaustive mask replacement map

The supplied mask is layer **14**, query head **8** (zero-based).
Use its original `layer_14/swap` experiment directory to inherit the exact
model, prompt/answer sample, seed, ratio, stride, selector, RoPE and background.
The script checks prompt/answer identity, target mask, initial scores and baseline
loss against that run. It does not regenerate a different synthetic sample.

## Run on the CUDA server

From the repository root, with the same environment as the original observation:

```bash
conda activate fyc_qwen
bash experiments2/run_mask_score.sh \
  output/ruler_observation/vt_32k_seed42_q220_layers/layer_14/swap \
  output/ruler_observation/mask_score_layer14_head8
```

If your actual original run is stored elsewhere, replace the first directory.
The directory must contain `experiment.json` and `block_map.npz`; its saved data
and model paths must still exist. Explicit `--model` or `--data` options may
override moved paths, but the experiment identity checks must still pass.
For an interrupted sweep, run the same command with `--resume` at the end.
Original results and `mask.png` are never overwritten.

Without a source run, the explicit equivalent is:

```bash
python -u experiments2/run_mask_score.py \
  --model /inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B \
  --data output/ruler_observation/vt_32k_seed42_q220_layers/_vt_data/observation.jsonl \
  --sample-index 0 --layer 14 --head 8 --query-block last \
  --ratio 0.65 --stride 8 --selector initial --background sparse \
  --seed 42 --output output/ruler_observation/mask_score_layer14_head8
```

## Exact trial definition

For each query-block row with both selected and unselected causally valid blocks,
remove the selected block with the lowest **initial** score (ties: smallest key
index). Independently substitute each originally unselected valid block once.
Each trial starts from the original baseline masks, changes exactly two entries
in one row of layer 14/head 8, preserves its retained-block count and freezes
every other layer/head/row selection. A new prefill and teacher-forced answer
decode recompute model states and KV from scratch.

Loss is the mean reference-answer token NLL: prompt tokens and extra EOS are
excluded, and the correct preceding answer tokens are supplied during decode.
Orange requires `replacement_loss < baseline_loss - tolerance`, where the final
tolerance is `max(--loss-tolerance, 5 * baseline_repeat_drift)`; default requested
tolerance is 1e-5. Equality and improvements smaller than that remain white.
Blue is always the **original** selected set, even for a block removed in a trial;
orange denotes a better alternative, not an accumulated optimized selection.
Grey is causally invalid future blocks, which are never tested.

This map measures answer-loss effects, not an attention-weight heatmap. Rows
with no legal white blocks require no trials. Earlier rows may have negligible
answer-loss effects, especially in later layers; this is an experimental result,
not evidence that the replacement implementation failed.

## Files and cost

The number of full model evaluations is the total number of causally valid white
cells plus baseline and repeat-baseline evaluations (about 11,000--12,000 trials
for 256 rows at a 65% retained budget). Every trial is flushed to `trials.jsonl`.
Resume checks the baseline mask fingerprint, settings, tokens and baseline loss,
and repairs only an interrupted final JSONL record. Do not run two writers on the
same output directory. Plotting requires a completed sweep so white never hides
untested candidates.

Outputs: `experiment.json`, `block_map.npz`, `trials.jsonl`, `trials.csv`,
`mask_score.npz` (color classes and tested mask), and
**`mask_score.png` / `mask_score.pdf` / `mask_score.svg`**.

To replot without GPU/model loading:

```bash
python experiments2/run_mask_score.py --plot-only \
  --output output/ruler_observation/mask_score_layer14_head8
```

This repository addition has CPU tests; real CUDA inference must be run on the
server. No measured orange map is provided before that experiment completes.

## Larger-font replot (no model or GPU required)

```bash
python experiments2/replot_mask_score.py \
  --input-dir output/ruler_observation/mask_score_layer14_head8
```

The independent script reads the completed experiment's metadata and saved
color classes (or reconstructs them from the baseline mask and trial log).
It preserves every block's class and writes `mask_score_large_fonts.png`,
`.pdf`, and `.svg` into `./acl_revision/Figs`, without overwriting the original result.
Default title/axis/tick/legend sizes are 22/20/17/15 pt; edit the constants
at the top of the script or use `--font-scale 1.1` to enlarge them further.
Use `--output-prefix figures/mask_score_large_fonts` to choose another
destination; the prefix should not include an extension.
