"""Run: python -m unittest discover -s experiments/block_label_swap -v"""
import unittest
import numpy as np
from swap_core import make_plan, swapped_mask, summarize


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[1, 0, 0, 0], [.8, .2, 0, 0],
                                [.6, .3, .1, 0], [.5, .1, .3, .1]])
        self.mask = np.array([[1, 0, 0, 0], [1, 0, 0, 0],
                              [1, 0, 1, 0], [1, 0, 1, 0]], dtype=bool)

    def test_remove_minimum_selected_not_global_minimum(self):
        plan = make_plan(self.scores, self.mask, 3)
        self.assertEqual(plan["removed_key_block"], 2)
        self.assertEqual(plan["candidate_key_blocks"], [1, 3])

    def test_candidates_exclude_future_even_when_future_has_low_score(self):
        plan = make_plan(self.scores, self.mask, 2)
        self.assertEqual(plan["candidate_key_blocks"], [1])
        self.assertEqual(plan["removed_key_block"], 2)

    def test_independent_swaps_preserve_other_rows_and_budget(self):
        plan = make_plan(self.scores, self.mask, 3)
        original = self.mask.copy()
        for candidate in plan["candidate_key_blocks"]:
            result = swapped_mask(self.mask, plan, candidate)
            np.testing.assert_array_equal(result.sum(1), self.mask.sum(1))
            np.testing.assert_array_equal(result[:3], self.mask[:3])
            self.assertFalse(result[3, 2])
            self.assertTrue(result[3, candidate])
        np.testing.assert_array_equal(self.mask, original)

    def test_tie_break_is_deterministic(self):
        self.scores[3, 0] = self.scores[3, 2]
        self.assertEqual(make_plan(self.scores, self.mask, 3)["removed_key_block"], 0)

    def test_reject_invalid_candidate(self):
        plan = make_plan(self.scores, self.mask, 2)
        with self.assertRaises(ValueError):
            swapped_mask(self.mask, plan, 3)

    def test_observation_requires_strictly_lower_score_and_lower_loss(self):
        rows = [{"key_block": 1, "initial_score": .1, "label_loss": .8},
                {"key_block": 3, "initial_score": .5, "label_loss": .5}]
        result = summarize(1., 1., rows, .3, 1e-5)
        self.assertEqual(result["lower_score_better_loss_count"], 1)

    def test_repeat_noise_not_claimed_as_evidence(self):
        rows = [{"key_block": 1, "initial_score": .1, "label_loss": .9999}]
        result = summarize(1., 1.0001, rows, .3, 1e-5)
        self.assertFalse(result["has_counterexample"])

    def test_worse_results_are_reported_honestly(self):
        rows = [{"key_block": 1, "initial_score": .1, "label_loss": 1.2}]
        self.assertFalse(summarize(1., 1., rows, .3, 1e-5)["has_counterexample"])

    def test_reject_row_without_candidates(self):
        with self.assertRaises(ValueError):
            make_plan(self.scores, self.mask, 0)


if __name__ == "__main__":
    unittest.main()
