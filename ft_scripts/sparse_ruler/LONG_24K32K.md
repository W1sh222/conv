# Sparse guarded 24K-32K training

Run from the repository root:

```bash
bash ft_scripts/run_ruler_mix_sparse_guarded.sh
```

The default is BF16 frozen-model weights/compute for an H200. The trainable
7x7 convolution and teacher accumulators remain FP32.

For a genuinely small GPU allocation, use NF4 storage with BF16 compute:

```bash
MODEL_PRECISION=nf4 bash ft_scripts/run_ruler_mix_sparse_guarded.sh
```

NF4 needs less memory but can introduce a Q/K distribution mismatch when
evaluation loads the language model in BF16. Prefer BF16 whenever it fits.

Before a full run, use a separate smoke-test output:

```bash
DATA_SAMPLES=64 \
TRAIN_STEPS=2 \
SAVE_STEPS=1 \
OUT_PATH=/tmp/conv_sparse_24k32k_smoke.pt \
bash ft_scripts/run_ruler_mix_sparse_guarded.sh
```

Resume an interrupted formal run with the saved training-state file:

```bash
RESUME_STATE=/absolute/path/to/checkpoint_train_state.pt \
bash ft_scripts/run_ruler_mix_sparse_guarded.sh
```

The long trainer does not right-truncate an over-length sample after an OOM,
because doing so would remove the final RULER question and silently train on
the wrong prompt.
