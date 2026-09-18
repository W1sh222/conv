import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from context_core import (causal_neighborhood, load_run, match_by_center,
                          residualize_by_center, summarize)


class ContextExperimentTests(unittest.TestCase):
    def test_neighborhood_excludes_center_future_and_out_of_bounds(self):
        scores = np.arange(25, dtype=float).reshape(5, 5)
        values, coordinates = causal_neighborhood(scores, 4, 3, 3)
        self.assertNotIn((4, 3), coordinates)
        self.assertTrue(all(k <= q for q, k in coordinates))
        self.assertTrue(all(0 <= q < 5 and 0 <= k < 5 for q, k in coordinates))
        np.testing.assert_array_equal(values, [scores[q, k] for q, k in coordinates])

    def test_matching_is_outcome_blind(self):
        records = self.synthetic_records()
        first = [(p["low_key_block"], p["high_key_block"])
                 for p in match_by_center(records, 1.0)]
        for index, record in enumerate(records):
            record["utility"] = 1000 * (-1) ** index
            record["replacement_loss"] = -record["utility"]
        second = [(p["low_key_block"], p["high_key_block"])
                  for p in match_by_center(records, 1.0)]
        self.assertEqual(first, second)

    def test_summary_detects_synthetic_positive_context_effect(self):
        records = self.synthetic_records(40)
        pairs = match_by_center(records, 1.0)
        result, residual = summarize(records, pairs, permutations=1000,
                                     bootstrap_samples=1000, seed=7)
        self.assertGreater(result["mean_utility_difference_high_minus_low_context"], 0)
        self.assertGreater(result["partial_neighbor_pearson_r"], .9)
        self.assertEqual(len(residual), len(records))

    def test_residualization_removes_linear_center_effect(self):
        records = self.synthetic_records()
        residual = residualize_by_center(records)
        center = np.array([r["center_score"] for r in residual])
        utility_residual = np.array([r["utility_residual"] for r in residual])
        self.assertAlmostEqual(float(np.corrcoef(center, utility_residual)[0, 1]), 0, places=10)

    def test_complete_sweep_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            scores = np.array([[.5, 0, 0], [.4, .3, 0], [.2, .1, .15]])
            mask = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=bool)
            meta = {"status": "complete", "arguments": {"layer": 1, "head": 0},
                    "plan": {"query_block": 2, "removed_key_block": 2,
                             "candidate_key_blocks": [1]},
                    "baseline": {"label_loss": 1.0},
                    "summary": {"effective_improvement_tolerance": 1e-5}}
            (path / "experiment.json").write_text(json.dumps(meta), encoding="utf-8")
            (path / "trials.jsonl").write_text(json.dumps(
                {"key_block": 1, "initial_score": .1, "label_loss": .9}) + "\n", encoding="utf-8")
            np.savez(path / "block_map.npz", initial_scores=scores, selected_mask=mask)
            run = load_run(path, kernel_size=3)
            self.assertEqual(len(run["records"]), 1)
            (path / "trials.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every candidate"):
                load_run(path, kernel_size=3)

    @staticmethod
    def synthetic_records(count=20):
        centers = np.linspace(0.0, 1.0, count)
        # Alternating context makes both groups cover the same center-score range.
        contexts = np.array([(i % 2) + .01 * i for i in range(count)], dtype=float)
        utilities = .02 * centers + .2 * contexts
        return [{"run_id": "run", "key_block": i, "center_score": float(centers[i]),
                 "neighbor_mean": float(contexts[i]), "utility": float(utilities[i]),
                 "replacement_loss": float(1 - utilities[i]), "effective_tolerance": 1e-8}
                for i in range(count)]


if __name__ == "__main__":
    unittest.main()
