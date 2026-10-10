"""CPU checks for the exhaustive sweep's mask and recovery semantics."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from run_mask_score import classify, read_trials, make_plan, plot_mask
from swap_core import swapped_mask


class MaskScoreTests(unittest.TestCase):
    def test_every_legal_white_cell_is_planned_once(self):
        mask = np.tri(5, dtype=bool)
        mask[2, 0] = mask[3, 1] = mask[4, 0] = False
        scores = np.arange(25).reshape(5, 5)
        plans = [make_plan(scores, mask, q) for q in (2, 3, 4)]
        self.assertEqual([(p['query_block'], k) for p in plans for k in p['candidate_key_blocks']],
                         [(2, 0), (3, 1), (4, 0)])
        before = mask.copy()
        for plan in plans:
            after = swapped_mask(mask, plan, plan['candidate_key_blocks'][0])
            np.testing.assert_array_equal(after.sum(1), mask.sum(1))
            np.testing.assert_array_equal(mask, before)
            self.assertEqual(np.count_nonzero(after != before), 2)

    def test_colors_preserve_blue_and_causal_grey(self):
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        rows = [{'query_block': 1, 'key_block': 0, 'delta_loss': -.01},
                {'query_block': 2, 'key_block': 0, 'delta_loss': 0.0},
                {'query_block': 2, 'key_block': 1, 'delta_loss': -.000001}]
        colors, tested = classify(mask, rows, .00001)
        np.testing.assert_array_equal(colors, [[1, 3, 3], [2, 1, 3], [0, 0, 1]])
        self.assertEqual(tested.sum(), 3)

    def test_duplicate_or_selected_or_future_trials_rejected(self):
        mask = np.eye(2, dtype=bool)
        good = {'query_block': 1, 'key_block': 0, 'delta_loss': -.1}
        for rows in ([good, good], [{'query_block': 0, 'key_block': 0, 'delta_loss': -.1}],
                     [{'query_block': 0, 'key_block': 1, 'delta_loss': -.1}]):
            with self.assertRaises(ValueError):
                classify(mask, rows, 0)

    def test_resume_repairs_only_truncated_final_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'trials.jsonl'
            path.write_bytes(b'{"query_block": 1}\n{"query_')
            self.assertEqual(read_trials(path, repair=True), [{'query_block': 1}])
            self.assertEqual(path.read_bytes(), b'{"query_block": 1}\n')
            path.write_bytes(b'broken\n{"query_block": 1}\n')
            with self.assertRaises(ValueError):
                read_trials(path, repair=True)

    def test_plot_exports_and_refuses_incomplete_sweep(self):
        # Tiny synthetic software fixture, removed on exit; never research data.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask = np.array([[1, 0], [0, 1]], dtype=bool)
            np.savez_compressed(root / 'block_map.npz', selected_mask=mask)
            meta = {'status': 'running', 'total_trials': 1,
                    'effective_improvement_tolerance': 1e-5,
                    'arguments': {'layer': 14, 'head': 8}}
            (root / 'experiment.json').write_text(json.dumps(meta), encoding='utf-8')
            with self.assertRaises(ValueError):
                plot_mask(root)
            meta['status'] = 'complete'
            (root / 'experiment.json').write_text(json.dumps(meta), encoding='utf-8')
            (root / 'trials.jsonl').write_text(json.dumps(
                {'query_block': 1, 'key_block': 0, 'delta_loss': -.1}) + '\n', encoding='utf-8')
            plot_mask(root)
            for extension in ('png', 'pdf', 'svg', 'npz'):
                self.assertTrue((root / f'mask_score.{extension}').exists())


if __name__ == '__main__':
    unittest.main()
