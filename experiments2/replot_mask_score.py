"""Replot a completed mask-score sweep with larger fonts, without GPU/Torch.

Example:
    python experiments2/replot_mask_score.py --input-dir \
        output/ruler_observation/mask_score_layer14_head8

Only display settings change; original experiment files remain untouched.
"""
from pathlib import Path
import argparse
import json
import numpy as np

# Edit these values to change the typography. Sizes are points.
TITLE_SIZE = 22
AXIS_LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 15
FIGURE_SIZE = (7.6, 7.6)
PALETTE = ['#FFFFFF', '#8DBAD8', '#F2AA65', '#EEEEEE']


def load_classes(directory):
    meta = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
    if meta.get('status') != 'complete':
        raise ValueError('The sweep must be complete: untested cells cannot be labeled as no improvement.')
    cached = directory / 'mask_score.npz'
    if cached.exists():
        with np.load(cached, allow_pickle=False) as archive:
            colors = archive['color_classes'].copy()
            selected = archive['selected_mask'].astype(bool)
            tested = archive['tested_mask'].astype(bool)
    else:
        with np.load(directory / 'block_map.npz', allow_pickle=False) as archive:
            selected = archive['selected_mask'].astype(bool)
        nq, nk = selected.shape
        legal = np.arange(nk)[None, :] <= np.arange(nq)[:, None]
        colors = np.where(legal, np.where(selected, 1, 0), 3).astype(np.uint8)
        tested = np.zeros_like(selected)
        tolerance = float(meta['effective_improvement_tolerance'])
        with (directory / 'trials.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                trial = json.loads(line)
                q, k = trial['query_block'], trial['key_block']
                delta = float(trial['delta_loss'])
                if not (0 <= q < nq and 0 <= k < nk) or not np.isfinite(delta):
                    raise ValueError('Invalid trial coordinates or loss difference.')
                if not legal[q, k] or selected[q, k] or tested[q, k]:
                    raise ValueError('Duplicate or invalid replacement candidate.')
                tested[q, k] = True
                if delta < -tolerance:
                    colors[q, k] = 2
    if colors.ndim != 2 or colors.shape != selected.shape or colors.shape != tested.shape:
        raise ValueError('Inconsistent mask shapes.')
    nq, nk = colors.shape
    legal = np.arange(nk)[None, :] <= np.arange(nq)[:, None]
    candidates = legal & ~selected
    if not np.array_equal(tested, candidates) or int(tested.sum()) != meta['total_trials']:
        raise ValueError('Missing or inconsistent trial coverage.')
    if not np.isin(colors, [0, 1, 2, 3]).all():
        raise ValueError('Unknown color class.')
    if not (np.all(colors[~legal] == 3) and np.all(colors[legal & selected] == 1)
            and np.isin(colors[candidates], [0, 2]).all()):
        raise ValueError('Color classes disagree with the baseline causal mask.')
    return colors, meta


def render(directory, prefix, scale=1.0, dpi=600):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    colors, meta = load_classes(directory)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': TICK_SIZE * scale, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    fig = plt.figure(figsize=FIGURE_SIZE)
    width = .83
<<<<<<< HEAD
    ax = fig.add_axes([.14, .105, width, width * FIGURE_SIZE[0] / FIGURE_SIZE[1]])
=======
    ax = fig.add_axes([.14, .23, width, width * FIGURE_SIZE[0] / FIGURE_SIZE[1]])
>>>>>>> 31e89147590bc7beb7459bf713488aa48b6f8a81
    ax.imshow(colors, cmap=ListedColormap(PALETTE),
              norm=BoundaryNorm(np.arange(-.5, 4.5), 4),
              interpolation='nearest', origin='upper', aspect='equal')
    ax.set_xlabel('Key block', fontsize=AXIS_LABEL_SIZE * scale, labelpad=10)
    ax.set_ylabel('Query block', fontsize=AXIS_LABEL_SIZE * scale, labelpad=10)
    args = meta['arguments']
    ax.set_title(f'Layer {args["layer"]} / Head {args["head"]}',
                 fontsize=TITLE_SIZE * scale, pad=14)
    for setter, n in [(ax.set_xticks, colors.shape[1]), (ax.set_yticks, colors.shape[0])]:
        step = 50 if n >= 150 else max(1, int(np.ceil(n / 6)))
        setter(np.arange(0, n, step))
    ax.tick_params(axis='both', labelsize=TICK_SIZE * scale, width=1, length=5, pad=6)
    labels = ['Selected', 'Unselected: lower loss',
              'Unselected: no lower loss', 'Causally masked']
    handles = [Patch(facecolor=PALETTE[i], edgecolor='#777777', linewidth=1,
                     label=label) for i, label in zip([1, 2, 0, 3], labels)]
<<<<<<< HEAD
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98, .98),
              ncol=1, frameon=False, fontsize=LEGEND_SIZE * scale,
              handlelength=1.5, handletextpad=.65, labelspacing=.8,
              borderaxespad=.3)
=======
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .045),
               ncol=2, frameon=False, fontsize=LEGEND_SIZE * scale,
               columnspacing=1.15, handlelength=1.5, handletextpad=.65, labelspacing=.65)
>>>>>>> 31e89147590bc7beb7459bf713488aa48b6f8a81
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Save the final rendered plot rectangle for reproducible layout inspection.
    box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    audit = {'panel_count': 1, 'alignment': 'not applicable: single panel',
             'plot_rect_inches': [box.x0, box.y0, box.width, box.height],
             'font_sizes_pt': {'title': TITLE_SIZE * scale, 'axis': AXIS_LABEL_SIZE * scale,
                               'tick': TICK_SIZE * scale, 'legend': LEGEND_SIZE * scale},
             'shape': list(colors.shape), 'classes': {str(i): int((colors == i).sum()) for i in range(4)},
             'experiment_data_modified': False}
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(str(prefix) + '.' + extension, dpi=dpi)
    Path(str(prefix) + '.layout.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    plt.close(fig)
    return colors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
                        default=Path('output/ruler_observation/mask_score_layer14_head8'))
    parser.add_argument('--output-prefix', type=Path,
                        default=Path('acl_revision/Figs/mask_score_large_fonts'),
                        help='Output path without extension; defaults to acl_revision/Figs/mask_score_large_fonts')
    parser.add_argument('--font-scale', type=float, default=1.0,
                        help='Multiply all font sizes; default 1.0 already enlarges the original fonts')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    if args.font_scale <= 0 or args.dpi <= 0:
        parser.error('--font-scale and --dpi must be positive')
    prefix = args.output_prefix
    render(args.input_dir, prefix, args.font_scale, args.dpi)
    print(f'Saved {prefix}.png / .pdf / .svg (no model loaded).')


if __name__ == '__main__':
    main()
