"""Reproduce Top-p selector edge cases without loading a language model.

Runs the actual find_blocks_chunked function extracted from utils.py.
Uses PyTorch when available, otherwise a small NumPy API adapter (CPU only).
Optional --weight / --scores inspect server checkpoints or saved score maps.
This script does not alter inference code, masks, or previous results.
"""
from pathlib import Path
import argparse
import ast
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class NumpyTensor(np.ndarray):
    @property
    def device(self):
        return 'cpu'

    def to(self, target=None):
        return wrap(np.asarray(self).astype(float) if target is float else self)

    def unsqueeze(self, dim):
        return wrap(np.expand_dims(np.asarray(self), dim))

    def expand(self, *shape):
        return wrap(np.broadcast_to(np.asarray(self), shape))

    def sum(self, dim=None, keepdim=False):
        return wrap(np.asarray(self).sum(axis=dim, keepdims=keepdim))

    def cumsum(self, dim):
        return wrap(np.asarray(self).cumsum(axis=dim))

    def masked_fill(self, mask, value):
        return wrap(np.where(mask, value, np.asarray(self)))

    def view(self, *shape):
        return wrap(np.asarray(self).reshape(shape))

    def clamp(self, lo, hi):
        return wrap(np.clip(np.asarray(self), lo, hi))


def wrap(value):
    return np.asarray(value).view(NumpyTensor)


class NumpyTorch:
    Tensor = NumpyTensor
    bool = bool

    def tensor(self, value):
        return wrap(value)

    def zeros(self, shape, device=None, dtype=float):
        return wrap(np.zeros(shape, dtype=dtype))

    def ones(self, *shape, device=None, dtype=float):
        return wrap(np.ones(shape, dtype=dtype))

    def zeros_like(self, x, dtype=None, device=None):
        return wrap(np.zeros_like(np.asarray(x), dtype=dtype))

    def ones_like(self, x, dtype=None):
        return wrap(np.ones_like(np.asarray(x), dtype=dtype))

    def eye(self, size, device=None):
        return wrap(np.eye(size))

    def arange(self, size, device=None):
        return wrap(np.arange(size))

    def tril(self, x):
        return wrap(np.tril(np.asarray(x)))

    def where(self, condition, yes, no):
        return wrap(np.where(condition, yes, no))

    def cat(self, arrays, dim):
        return wrap(np.concatenate(arrays, axis=dim))

    def sort(self, x, dim, descending=False):
        values = np.asarray(x)
        indices = np.argsort(-values if descending else values, axis=dim, kind='stable')
        return wrap(np.take_along_axis(values, indices, dim)), wrap(indices)


def load_selector():
    try:
        import torch
        backend = 'PyTorch CPU'
    except ImportError:
        torch = NumpyTorch()
        backend = 'NumPy API adapter; no CUDA or model inference'
    tree = ast.parse((ROOT / 'xattn/src/utils.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'find_blocks_chunked')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'xattn/src/utils.py', 'exec'), ns)
    return torch, ns['find_blocks_chunked'], backend


def reference_positive_mass(scores, offset, p):
    """Diagnostic reference: causal nonnegative mass, forced sink+diagonal.

    Clamping here only defines a diagnostic mass; it is not a proposed final
    calibration policy for trained Conv scores.
    """
    b, h, q, k = scores.shape
    valid = np.arange(k)[None, :] <= (offset + np.arange(q))[:, None]
    mass = np.where(valid[None, None], np.maximum(scores, 0), 0)
    mask = np.zeros_like(mass, dtype=bool)
    for bi in range(b):
        for hi in range(h):
            for qi in range(q):
                row = mass[bi, hi, qi]
                selected = mask[bi, hi, qi]
                selected[0] = True
                selected[offset + qi] = True
                target = row.sum() * p
                retained = row[selected].sum()
                for ki in np.argsort(-row, kind='stable'):
                    if retained >= target:
                        break
                    if not selected[ki]:
                        selected[ki] = True
                        retained += row[ki]
    return mask & valid[None, None]


def evaluate(scores, offset, p, torch, selector):
    tensor = torch.tensor(scores)
    raw = selector(tensor, offset, p, None, decoding=False, mode='prefill', causal=True)
    raw = raw.detach().cpu().numpy() if hasattr(raw, 'detach') else np.asarray(raw)
    q, k = scores.shape[-2:]
    valid = np.arange(k)[None, :] <= (offset + np.arange(q))[:, None]
    # Match the final causal cleanup in Conv.py.
    mask = raw & valid[None, None]
    mass = np.where(valid[None, None], np.maximum(scores, 0), 0)
    total = mass.sum(-1)
    retained = (mass * mask).sum(-1)
    coverage = np.divide(retained, total, out=np.ones_like(total), where=total > 0)
    reference = reference_positive_mass(scores, offset, p)
    result = {
        'offset': offset, 'p': p,
        'legacy_kept': np.flatnonzero(mask[0, 0, -1]).tolist(),
        'causal_positive_mass_coverage_last_row': float(coverage[0, 0, -1]),
        'reference_kept': np.flatnonzero(reference[0, 0, -1]).tolist(),
        'rows_below_requested_positive_mass': int((coverage + 1e-10 < p).sum()),
        'negative_score_fraction': float((scores < 0).mean()),
        'nonzero_future_score_entries': int(((scores != 0) & ~valid[None, None]).sum()),
        'legacy_mean_causal_density': float(mask.sum() / (valid.sum() * scores.shape[0] * scores.shape[1])),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', type=Path)
    parser.add_argument('--scores', type=Path, help='Post-convolution score map .npy or tensor .pt, shape [B,H,Q,K]')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--topp', type=float, default=.95)
    parser.add_argument('--output', type=Path, default=ROOT/'output/ruler_topp_audit/report.json')
    args = parser.parse_args()
    torch, selector, backend = load_selector()
    cases = [
        ('nonnegative_causal_control', [.1, .3, .2, .15, .1, .05, .04, .06], 7),
        ('negative_mass_cancellation', [.1, .5, .25, .15, .05, -.3, -.25, .1], 7),
        ('future_mass_removed_after_selection', [.1, .2, .3, .4, 20., 0, 0, 0], 3),
        ('peaked_mass_is_not_95_percent_blocks', [.005, .97, .005, .005, .005, 0, 0, .01], 7),
    ]
    results = {}
    for name, row, offset in cases:
        scores = np.array(row, dtype=float)[None, None, None]
        results[name] = evaluate(scores, offset, args.topp, torch, selector)
    assert results['nonnegative_causal_control']['rows_below_requested_positive_mass'] == 0
    if args.topp == .95:
        assert results['negative_mass_cancellation']['rows_below_requested_positive_mass'] == 1
        assert results['future_mass_removed_after_selection']['rows_below_requested_positive_mass'] == 1
    report = {'backend': backend, 'synthetic_counterexamples': results,
              'scope': 'Selector-level checks only; these do not establish the cause of a particular model run.'}
    if args.weight or (args.scores and args.scores.suffix != '.npy'):
        if isinstance(torch, NumpyTorch):
            raise RuntimeError('Inspect .pt files on the evaluation host with PyTorch installed.')
    if args.weight:
        weight = torch.load(args.weight, map_location='cpu', weights_only=True).float()
        report['weight'] = {'path': str(args.weight), 'shape': list(weight.shape),
                            'min': float(weight.min()), 'max': float(weight.max()),
                            'negative_fraction': float((weight < 0).float().mean())}
    if args.scores:
        if args.scores.suffix == '.npy':
            scores = np.load(args.scores)
        else:
            scores = torch.load(args.scores, map_location='cpu', weights_only=True).double().numpy()
        report['actual_scores'] = evaluate(scores, args.offset, args.topp, torch, selector)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
