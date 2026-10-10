#!/usr/bin/env python3
"""Exhaustive, independent per-row swaps; orange marks lower answer NLL.

Reuses the original observation runner's mask controller and label_nll.
GPU evaluation requires the server's Transformers 4.51/block_sparse environment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'block_label_swap'))
import run_experiment as original
from run_experiment import MaskController, label_nll, read_example, prepare_tokens
from swap_core import make_plan


def parse_args():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--source-run')
    preliminary.add_argument('--plot-only', action='store_true')
    known, _ = preliminary.parse_known_args()
    p = original.parser()
    p.description = __doc__
    for action in p._actions:
        if action.dest == 'query_block':
            action.help = 'Baseline recording row only; the sweep always tests ALL eligible rows'
    p.add_argument('--source-run', help='Original layer_14/swap directory: inherit exact experiment settings')
    p.add_argument('--resume', action='store_true', help='Continue only missing row/candidate trials')
    p.add_argument('--plot-only', action='store_true', help='Render completed output without loading a model')
    if known.source_run:
        meta = json.loads((Path(known.source_run) / 'experiment.json').read_text(encoding='utf-8'))
        defaults = {k: v for k, v in meta['arguments'].items() if k not in ('output', 'no_plots')}
        p.set_defaults(**defaults)
        for action in p._actions:
            if action.dest in defaults:
                action.required = False
    if known.plot_only:
        for action in p._actions:
            if action.dest != 'output':
                action.required = False
    return p.parse_args()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def read_trials(path, repair=False):
    if not path.exists():
        return []
    raw = path.read_bytes()
    rows, offset = [], 0
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if not repair or index != len(lines) - 1:
                raise
            # A killed process can leave only its final record truncated.
            with path.open('r+b') as f:
                f.truncate(offset)
            break
        rows.append(row)
        offset += len(line)
    if repair and offset and offset == len(raw) and not raw.endswith(b'\n'):
        with path.open('ab') as f:
            f.write(b'\n')
    return rows


def classify(selected, rows, tolerance):
    selected = np.asarray(selected, dtype=bool)
    n, m = selected.shape
    legal = np.arange(m)[None, :] <= np.arange(n)[:, None]
    colors = np.where(legal, np.where(selected, 1, 0), 3).astype(np.uint8)
    tested = np.zeros_like(selected)
    for row in rows:
        q, k = row['query_block'], row['key_block']
        if not legal[q, k] or selected[q, k] or tested[q, k]:
            raise ValueError('Trial must target a distinct, causally valid baseline-white block')
        tested[q, k] = True
        if row['delta_loss'] < -tolerance:
            colors[q, k] = 2
    return colors, tested


def plot_mask(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    meta = json.loads((out / 'experiment.json').read_text(encoding='utf-8'))
    if meta['status'] != 'complete':
        raise ValueError('Plotting requires a complete sweep: untested white cells must not imply no improvement')
    selected = np.load(out / 'block_map.npz')['selected_mask']
    rows = read_trials(out / 'trials.jsonl')
    colors, tested = classify(selected, rows, meta['effective_improvement_tolerance'])
    if len(rows) != meta['total_trials'] or int(tested.sum()) != meta['total_trials']:
        raise ValueError('Incomplete sweep')
    np.savez_compressed(out / 'mask_score.npz', color_classes=colors, tested_mask=tested,
                        selected_mask=selected)
    palette = ['#FFFFFF', '#8DBAD8', '#F2AA65', '#EEEEEE']
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(6.8, 7.3))
    fig.subplots_adjust(left=.12, right=.98, top=.92, bottom=.16)
    ax.imshow(colors, cmap=ListedColormap(palette), norm=BoundaryNorm(np.arange(-.5, 4.5), 4),
              interpolation='nearest', origin='upper', aspect='equal')
    ax.set(xlabel='Key block', ylabel='Query block',
           title=f'Layer {meta["arguments"]["layer"]} / Head {meta["arguments"]["head"]}')
    labels = ['Selected', 'Unselected: lower loss', 'Unselected: no lower loss', 'Causally masked']
    handles = [Patch(facecolor=palette[i], edgecolor='#777777', label=label)
               for i, label in zip([1, 2, 0, 3], labels)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .025), ncol=2,
               frameon=False, fontsize=9)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'mask_score.{extension}', dpi=600)
    plt.close(fig)


def main():
    args = parse_args()
    if args.plot_only:
        plot_mask(Path(args.output))
        return
    if not 0 < args.ratio < 1:
        raise ValueError("ratio must be in (0,1) to leave replacement candidates")
    if args.stride <= 0 or 128 % args.stride:
        raise ValueError("stride must be a positive divisor of block size 128")
    if args.loss_tolerance < 0:
        raise ValueError("loss-tolerance must be nonnegative")
    if args.selector == "conv" and not args.conv_weights:
        raise ValueError("--selector conv requires --conv-weights; no implicit averaging kernel")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError("Use a new/empty output directory; previous results are not overwritten")
    if args.resume and not (out / "experiment.json").exists():
        raise ValueError("--resume requires an existing experiment.json")
    # Fail on missing plotting dependencies before expensive model evaluation.
    if not args.no_plots:
        import matplotlib  # noqa: F401
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA plus the repository's Triton/block_sparse_attn environment is required")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ft_scripts" / "sparse_ruler"))
    from xattn.src import load_transformers_451 as adapter
    from xattn.src import Conv as conv
    from conv_sparse_ops import kernels_conv_block_scores_infer_full
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type not in {"llama", "qwen3"}:
        raise ValueError("Only Llama/Qwen3 Transformers 4.51 interfaces are supported")
    if getattr(config, "use_sliding_window", False):
        raise ValueError("Sliding-window attention is not supported by this experiment")
    if not 0 <= args.layer < config.num_hidden_layers or not 0 <= args.head < config.num_attention_heads:
        raise ValueError("layer/head index outside model layout")
    if args.rope_factor is not None:
        if args.rope_factor <= 0:
            raise ValueError("rope-factor must be positive")
        config.rope_scaling = {"rope_type": "yarn", "factor": args.rope_factor,
                              "original_max_position_embeddings": args.rope_original_length}
        config.max_position_embeddings = math.ceil(args.rope_original_length * args.rope_factor)
    if args.max_position_embeddings:
        config.max_position_embeddings = args.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    example = read_example(args.data, args.sample_index)
    prompt, label, prompt_ids, label_ids = prepare_tokens(tokenizer, example, args)
    if len(prompt_ids) + len(label_ids) > config.max_position_embeddings:
        raise ValueError("Prompt+label exceeds configured context; set correct RoPE options explicitly")
    controller = MaskController(args, len(prompt_ids), torch, adapter, conv,
                                kernels_conv_block_scores_infer_full)
    dm = args.device_map if args.device_map in {"auto", "balanced", "balanced_low_0", "sequential"} else {"": args.device_map}
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config,
              torch_dtype=getattr(torch, args.dtype), device_map=dm, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    # SDPA lets Transformers avoid allocating the global N*N additive causal mask.
    fast_config = adapter.BaseFastPrefillConfig(metric="conv", stride=args.stride,
                          block_topk_ratio=args.ratio, print_detail=False)
    original_prefill = adapter._run_prefill
    originals = []
    adapter._run_prefill = controller.prefill
    for layer in model.model.layers:
        attn = layer.self_attn
        originals.append((attn, attn.forward, getattr(attn, "fastprefillconfig", None)))
        attn.fastprefillconfig = fast_config
        attn.forward = adapter.forward_eval_451.__get__(attn, type(attn))
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        baseline = label_nll(model, prompt_ids, label_ids, torch)
        controller.recording = False
        selected = controller.frozen_masks[args.layer][0, args.head].numpy().copy()
        scores = controller.initial_scores
        plans = []
        for qb in range(controller.nblocks):
            if selected[qb, :qb + 1].all() or not selected[qb, :qb + 1].any():
                continue
            plans.append(make_plan(scores, selected, qb))
        total = sum(len(p['candidate_key_blocks']) for p in plans)
        signature = hashlib.sha256()
        for layer_index, mask in sorted(controller.frozen_masks.items()):
            signature.update(str(layer_index).encode())
            signature.update(mask.numpy().tobytes())
        identity = {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'label_ids': label_ids, 'mask_sha256': signature.hexdigest(),
                    'settings': {k: getattr(args, k) for k in
                        ('model', 'layer', 'head', 'ratio', 'stride', 'selector',
                         'conv_weights', 'background', 'dtype', 'seed', 'rope_factor',
                         'rope_original_length', 'max_position_embeddings', 'loss_tolerance')}}
        meta_path = out / 'experiment.json'
        if args.resume and meta_path.exists():
            previous = json.loads(meta_path.read_text(encoding='utf-8'))
            if previous['identity'] != identity:
                raise ValueError('Resume configuration, tokens or baseline masks changed')
            if abs(previous['baseline']['label_loss'] - baseline['label_loss']) > max(args.loss_tolerance, 1e-7):
                raise ValueError('Baseline loss changed; use a new output directory')
            baseline = previous['baseline']
        if args.source_run:
            source = Path(args.source_run)
            old = json.loads((source / 'experiment.json').read_text(encoding='utf-8'))
            old_map = np.load(source / 'block_map.npz')
            if old['prompt_sha256'] != identity['prompt_sha256'] or old['label_ids'] != label_ids:
                raise ValueError('The source experiment uses different prompt/answer tokens')
            if not np.array_equal(old_map['selected_mask'], selected):
                raise ValueError('Target mask differs from the source mask.png experiment')
            if not np.allclose(old_map['initial_scores'], scores, rtol=1e-5, atol=1e-7):
                raise ValueError('Initial score map differs from the source experiment')
            if abs(old['baseline']['label_loss'] - baseline['label_loss']) > max(args.loss_tolerance, 1e-7):
                raise ValueError('Source baseline loss differs; check model/environment/configuration')
        np.savez_compressed(out / 'block_map.npz', initial_scores=scores, selected_mask=selected)
        meta = {'status': 'running', 'arguments': vars(args), 'identity': identity,
                'baseline': baseline, 'plans': plans, 'total_trials': total,
                'prompt_token_count': len(prompt_ids), 'label_token_count': len(label_ids),
                'loss_definition': 'Mean teacher-forced reference answer NLL; prompt/EOS excluded',
                'scope': 'One row and one block swap per independent full model trial; all other masks fixed',
                'color_definition': 'Blue=baseline selected; orange=unselected with loss improvement above tolerance; white=other unselected; grey=future'}
        write_json(meta_path, meta)
        trials_path = out / 'trials.jsonl'
        rows = read_trials(trials_path, repair=args.resume)
        allowed = {(p['query_block'], c) for p in plans for c in p['candidate_key_blocks']}
        done = set()
        for row in rows:
            key = (row['query_block'], row['key_block'])
            if key not in allowed or key in done or not math.isfinite(row['label_loss']):
                raise ValueError('Invalid or duplicate saved trial')
            done.add(key)
        print(f'Baseline={baseline["label_loss"]:.9f}; rows={len(plans)}; trials={total}; completed={len(done)}', flush=True)
        with trials_path.open('a', encoding='utf-8') as log:
            for plan in plans:
                controller.plan = plan
                for candidate in plan['candidate_key_blocks']:
                    if (plan['query_block'], candidate) in done:
                        continue
                    controller.replacement = candidate
                    result = label_nll(model, prompt_ids, label_ids, torch)
                    row = {'query_block': plan['query_block'], 'key_block': candidate,
                           'removed_key_block': plan['removed_key_block'],
                           'initial_score': float(scores[plan['query_block'], candidate]),
                           'removed_initial_score': plan['removed_initial_score'],
                           'baseline_loss': baseline['label_loss'],
                           'delta_loss': result['label_loss'] - baseline['label_loss'], **result}
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    rows.append(row)
                    done.add((plan['query_block'], candidate))
                    print(f'[{len(done)}/{total}] q={plan["query_block"]}, {plan["removed_key_block"]}->{candidate}, delta={row["delta_loss"]:+.9f}', flush=True)
        controller.replacement = None
        repeated = label_nll(model, prompt_ids, label_ids, torch)
        drift = abs(repeated['label_loss'] - baseline['label_loss'])
        tolerance = max(args.loss_tolerance, 5 * drift)
        improved = sum(row['delta_loss'] < -tolerance for row in rows)
        meta.update(status='complete', repeated_baseline=repeated,
                    effective_improvement_tolerance=tolerance, baseline_repeat_drift=drift,
                    completed_trials=len(rows), improved_count=improved,
                    improved_fraction=improved / total if total else None,
                    elapsed_seconds=time.time() - start)
        write_json(meta_path, meta)
        fields = ['query_block', 'key_block', 'removed_key_block', 'initial_score',
                  'removed_initial_score', 'baseline_loss', 'label_loss', 'delta_loss']
        with (out / 'trials.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        if not args.no_plots:
            plot_mask(out)
        print(f'Complete: {improved}/{total} improving replacements; output={out}', flush=True)
    finally:
        adapter._run_prefill = original_prefill
        for attn, forward, old_config in originals:
            attn.forward = forward
            if old_config is None:
                delattr(attn, 'fastprefillconfig')
            else:
                attn.fastprefillconfig = old_config


if __name__ == "__main__":
    main()
