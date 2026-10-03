"""Selector invariants; uses PyTorch if installed, otherwise a NumPy adapter.

Run: python experiments/test_conv_topp_selector.py
No model, checkpoint, CUDA, or evaluation output is required.
"""
import ast
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
import numpy as np
from audit_ruler_topp import NumpyTensor, NumpyTorch, wrap

ROOT = Path(__file__).resolve().parents[1]
INVARIANTS_ONLY = '--invariants-only' in sys.argv
if INVARIANTS_ONLY:
    sys.argv.remove('--invariants-only')


class Adapter(NumpyTorch):
    float32 = np.float32
    long = np.int64

    def is_tensor(self, x):
        return isinstance(x, NumpyTensor)
    def isfinite(self, x):
        return wrap(np.isfinite(np.asarray(x)))
    def nan_to_num(self, x, **kw):
        return wrap(np.nan_to_num(np.asarray(x), **kw))
    def sort(self, x, dim=-1, descending=False, stable=False):
        return super().sort(x, dim, descending)
    def ceil(self, x):
        return wrap(np.ceil(np.asarray(x)))
    def full(self, shape, fill_value, dtype=float, device=None):
        return wrap(np.full(shape, fill_value, dtype=dtype))
    def ones(self, *shape, dtype=float, device=None):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return wrap(np.ones(shape, dtype=dtype))
    def zeros(self, *shape, dtype=float, device=None):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return wrap(np.zeros(shape, dtype=dtype))
    def topk(self, x, k, dim=-1):
        values, indices = self.sort(x, dim, True)
        sl = [slice(None)]*x.ndim
        sl[dim] = slice(k)
        return SimpleNamespace(values=values[tuple(sl)], indices=indices[tuple(sl)])
    def tril(self, x, diagonal=0):
        return wrap(np.tril(np.asarray(x), k=diagonal))


def scatter(self, dim, index, src):
    np.put_along_axis(self, np.asarray(index), np.broadcast_to(src, index.shape), axis=dim)
    return self


def adapt_numpy():
    NumpyTensor.float = lambda x: wrap(np.asarray(x, dtype=np.float32))
    NumpyTensor.long = lambda x: wrap(np.asarray(x, dtype=np.int64))
    NumpyTensor.clamp_min = lambda x, lo: wrap(np.maximum(np.asarray(x), lo))
    NumpyTensor.clamp = lambda x, lo=None, hi=None, **kw: wrap(np.clip(np.asarray(x), kw.get('min', lo), kw.get('max', hi)))
    NumpyTensor.contiguous = lambda x: wrap(np.ascontiguousarray(x))
    NumpyTensor.numel = lambda x: x.size
    NumpyTensor.to = lambda x, target=None, **kw: wrap(np.asarray(x, dtype=kw.get('dtype', None if isinstance(target,str) else target)))
    NumpyTensor.scatter_ = scatter
    NumpyTensor.expand = lambda x, *shape: wrap(np.broadcast_to(x, tuple(x.shape[i] if n == -1 else n for i, n in enumerate(shape))))


try:
    import torch
    BACKEND = 'PyTorch CPU'
except ImportError:
    adapt_numpy()
    torch = Adapter()
    BACKEND = 'NumPy API adapter (no PyTorch/model execution)'


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def load_source(text, names, ns):
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual_source_functions', 'exec'), ns)
    return ns


ns = {'torch': torch, 'math': math, 'os': os, '_LOGGED_POLICIES': set(), 'CONV_TOPP_VERSION': 'positive_mass_v1'}
load_source((ROOT/'xattn/src/conv_topp.py').read_text(encoding='utf-8-sig'),
            {'get_conv_topp_policy', 'select_conv_topp_blocks', 'select_conv_topp_mask'}, ns)
select = ns['select_conv_topp_blocks']
select_full = ns['select_conv_topp_mask']


class SelectorTests(unittest.TestCase):
    def check_mass(self, scores, p=.95, offset=0, causal=True):
        raw = np.asarray(scores, dtype=np.float32)
        mask = array(select(torch.tensor(raw), p, offset=offset, causal=causal))
        q, k = raw.shape[-2:]
        valid = np.arange(k)[None, :] <= offset+np.arange(q)[:, None] if causal else np.ones((q,k), bool)
        self.assertFalse(np.any(mask & ~valid[None,None]))
        mass = np.maximum(np.nan_to_num(raw, nan=0, posinf=1e4, neginf=-1e4), 0)*valid[None,None]
        total = mass.sum(-1)
        kept = (mass*mask).sum(-1)
        self.assertTrue(np.all(kept+1e-5 >= total*np.asarray(p)))
        zero = total <= 0
        self.assertTrue(np.all(mask[zero] == np.broadcast_to(valid[None,None], mask.shape)[zero]))
        return mask

    def test_signed_counterexample(self):
        mask = self.check_mass(np.array([.1,.5,.25,.15,.05,-.3,-.25,.1])[None,None,None], offset=7)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0,1,2,3,7])

    def test_future_counterexample(self):
        mask = self.check_mass(np.array([.1,.2,.3,.4,20,0,0,0])[None,None,None], offset=3)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0,1,2,3])

    def test_forced_mass_not_counted_twice(self):
        row = np.array([.48,.01,.01,.01,.01,.48])[None,None,None]
        mask = self.check_mass(row, offset=5)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0,5])

    def test_crossing_block_included(self):
        mask = self.check_mass(np.array([0,.5,.3,.2,0])[None,None,None], p=.7, offset=4)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0,1,2,4])

    def test_zero_and_all_negative_dense_fallback(self):
        for value in (0,-2):
            self.check_mass(np.full((2,3,6,6), value))

    def test_p_one_retains_all_valid_blocks(self):
        scores = np.random.default_rng(4).normal(size=(2,3,6,8))
        mask = self.check_mass(scores, p=1, offset=2)
        self.assertTrue(np.all(mask == (np.arange(8)[None,:] <= np.arange(6)[:,None]+2)))

    def test_monotone_masks_random_signed_scores(self):
        scores = np.random.default_rng(42).normal(size=(2,4,9,12))
        previous = np.zeros_like(scores, dtype=bool)
        for p in (.1,.5,.9,.95,1):
            mask = self.check_mass(scores, p=p, offset=3)
            self.assertFalse(np.any(previous & ~mask))
            previous = mask

    def test_padding_cannot_change_real_selection(self):
        real = np.random.default_rng(3).normal(size=(1,2,6,6)).astype(np.float32)
        padded = np.full((1,2,8,8), 1e6, dtype=np.float32)
        padded[...,:6,:6] = real
        expected = self.check_mass(real)
        for chunk in (1,4,8):
            mask = array(select_full(torch.tensor(padded), .95, 6, 6, chunk))
            np.testing.assert_array_equal(mask[...,:6,:6], expected)
            self.assertFalse(mask[...,6:,:].any())
            self.assertFalse(mask[...,:,6:].any())

    def test_noncausal_and_per_head_thresholds(self):
        scores = np.random.default_rng(3).uniform(size=(1,2,4,6)).astype(np.float32)
        self.check_mass(scores, causal=False)
        thresholds = torch.tensor([.5,.95])
        together = array(select(torch.tensor(scores), thresholds))
        for head, p in enumerate((.5,.95)):
            expected = array(select(torch.tensor(scores[:,head:head+1]), p))
            np.testing.assert_array_equal(together[:,head:head+1], expected)

    def test_invalid_thresholds(self):
        for p in (0,-1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                select(torch.tensor(np.ones((1,1,2,2))), p)

    @unittest.skipIf(INVARIANTS_ONLY, 'Requires the original repository HEAD for regression comparison')
    def test_training_uses_same_mask_and_topk_is_unchanged(self):
        path = 'ft_scripts/sparse_ruler/conv_sparse_ops.py'
        baseline = subprocess.check_output(['git','show','HEAD:'+path], cwd=ROOT).decode('utf-8-sig')
        current = (ROOT/path).read_text(encoding='utf-8-sig')
        funcs = {'make_inference_chunked_block_mask'}
        from audit_ruler_topp import load_selector
        _, legacy_selector, _ = load_selector()
        common = dict(ns, find_blocks_chunked=legacy_selector)
        old_ns, new_ns = load_source(baseline, funcs, dict(common)), load_source(current, funcs, dict(common))
        scores = torch.tensor(np.random.default_rng(42).normal(size=(1,2,8,8)).astype(np.float32))
        for ratio in (.3,.7,1):
            old = old_ns['make_inference_chunked_block_mask'](scores,.95,6,6,4,topk_ratio=ratio)
            new = new_ns['make_inference_chunked_block_mask'](scores,.95,6,6,4,topk_ratio=ratio)
            np.testing.assert_array_equal(array(old), array(new))
        old_policy = os.environ.pop('CONV_TOPP_SELECTOR', None)
        try:
            actual = new_ns['make_inference_chunked_block_mask'](scores,.95,6,6,4)
            expected = select_full(scores,.95,6,6,4)[...,:6,:6]
            np.testing.assert_array_equal(array(actual), array(expected))
            os.environ['CONV_TOPP_SELECTOR'] = 'legacy'
            actual = new_ns['make_inference_chunked_block_mask'](scores,.95,6,6,4)
            expected = old_ns['make_inference_chunked_block_mask'](scores,.95,6,6,4)
            np.testing.assert_array_equal(array(actual), array(expected))
        finally:
            if old_policy is None:
                os.environ.pop('CONV_TOPP_SELECTOR', None)
            else:
                os.environ['CONV_TOPP_SELECTOR'] = old_policy

    @unittest.skipIf(INVARIANTS_ONLY, 'Requires the original repository HEAD for regression comparison')
    def test_inference_topk_functions_and_original_branch_unchanged(self):
        path = 'xattn/src/Conv.py'
        baseline = ast.parse(subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT).decode('utf-8-sig'))
        current = ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
        for name in ('_topk_ratio_mask_from_scores','_fixed_topk_mask_from_scores','_safe_causal_topk_mask'):
            old = next(n for n in baseline.body if isinstance(n,ast.FunctionDef) and n.name == name)
            new = next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(old),ast.dump(new))
        def old_branch(tree):
            fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'conv_estimate')
            return next(n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test).startswith('conv_safe_topk'))
        self.assertEqual(ast.dump(old_branch(baseline)),ast.dump(old_branch(current)))

    @unittest.skipIf(INVARIANTS_ONLY, 'Requires the original repository HEAD for regression comparison')
    def test_actual_inference_dispatch_and_legacy_reproduction(self):
        path = 'xattn/src/Conv.py'
        old_text = subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT).decode('utf-8-sig')
        new_text = (ROOT/path).read_text(encoding='utf-8-sig')
        from audit_ruler_topp import load_selector
        _, legacy_selector, _ = load_selector()
        scores = torch.tensor(np.random.default_rng(17).normal(size=(1,2,8,8)).astype(np.float32))
        common = dict(ns, find_blocks_chunked=legacy_selector,
                      attn_sums=scores, attn_sums_smoothed=scores, q_len=6, k_len=6,
                      block_size=1, num_blocks_per_chunk=4, q_chunk_num=2,
                      q_block_num=8, k_block_num=8, threshold=.95,
                      causal=True, keep_sink=False, keep_recent=False, num_kv_head=2,
                      fallback_topk=2)
        helpers = {'_threshold_to_float','_safe_causal_topk_mask','_topk_ratio_mask_from_scores','_fixed_topk_mask_from_scores'}
        def dispatch(text, corrected, ratio=None, fixed=None, safe=False):
            env = load_source(text, helpers, dict(common, topk_ratio=ratio, fixed_topk=fixed, conv_safe_topk=safe))
            tree = ast.parse(text)
            fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'conv_estimate')
            if corrected:
                start = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'topp_policy')
            else:
                start = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('conv_safe_topk'))
            wrapper = ast.parse('def choose():\n    pass').body[0]
            wrapper.body = fn.body[start:]
            exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])), 'actual_inference_dispatch', 'exec'), env)
            return array(env['choose']()[1])
        old_policy = os.environ.pop('CONV_TOPP_SELECTOR', None)
        try:
            actual = dispatch(new_text, True)
            expected = array(select_full(scores,.95,6,6,4))
            np.testing.assert_array_equal(actual, expected)
            # Top-k must ignore even an invalid Top-p-specific environment value.
            os.environ['CONV_TOPP_SELECTOR'] = 'ignored_by_topk'
            for kw in ({'ratio':.7},{'fixed':3},{'safe':True}):
                np.testing.assert_array_equal(dispatch(old_text,False,**kw), dispatch(new_text,True,**kw))
            os.environ['CONV_TOPP_SELECTOR'] = 'legacy'
            np.testing.assert_array_equal(dispatch(old_text,False), dispatch(new_text,True))
        finally:
            if old_policy is None:
                os.environ.pop('CONV_TOPP_SELECTOR', None)
            else:
                os.environ['CONV_TOPP_SELECTOR'] = old_policy


if __name__ == '__main__':
    print('Backend:', BACKEND, flush=True)
    unittest.main(verbosity=2)
