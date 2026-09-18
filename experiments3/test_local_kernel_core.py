import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from local_kernel_core import (
    blocked_purged_folds,
    evaluate_cv,
    extract_replicate_patch,
    load_run,
    pooled_offset_correlations,
    prepare_cv_plans,
    prepare_offset_state,
)


class LocalKernelExperimentTests(unittest.TestCase):
    def test_patch_matches_replicate_padding_and_cross_correlation_orientation(self):
        scores = np.arange(16, dtype=float).reshape(4, 4)
        expected = np.pad(scores, 1, mode="edge")[1:4, 0:3]
        actual = extract_replicate_patch(scores, query_block=1, key_block=0, kernel_size=3)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[1, 1], scores[1, 0])

    def test_purged_folds_prevent_patch_overlap_along_key_axis(self):
        keys = np.arange(10, 100)
        for train, test in blocked_purged_folds(keys, folds=5, purge_radius=6):
            distances = np.abs(keys[train, None] - keys[None, test])
            self.assertTrue(np.all(distances > 6))

    def test_signed_local_model_recovers_positive_and_negative_offsets(self):
        run = self.synthetic_run()
        plans = prepare_cv_plans(run, folds=5, purge_radius=6, ridge_alpha=.01,
                                 control_mode="center_position", subset="full")
        result = evaluate_cv(run, plans, return_kernels=True)
        self.assertGreater(result["metrics"]["relative_mse_reduction"], .65)
        mean_kernel = np.mean([item["kernel"] for item in result["kernels"]], axis=0)
        # Target is +score[dq=-1,dk=+2] - score[dq=+1,dk=-2].
        self.assertGreater(mean_kernel[2, 5], 0)
        self.assertLess(mean_kernel[4, 1], 0)
        state = prepare_offset_state(run, "center_position")
        corr = pooled_offset_correlations([run], [state])
        self.assertGreater(corr[2, 5], .4)
        self.assertLess(corr[4, 1], -.4)

    def test_complete_swap_sweep_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            scores = np.arange(16, dtype=float).reshape(4, 4) / 10
            mask = np.zeros((4, 4), dtype=bool)
            mask[3, 3] = True
            meta = {
                "status": "complete",
                "arguments": {"layer": 2, "head": 1, "sample_index": 0, "data": "sample.jsonl"},
                "plan": {"query_block": 3, "removed_key_block": 3,
                         "candidate_key_blocks": [1, 2]},
                "baseline": {"label_loss": 1.0},
            }
            trials = [
                {"key_block": 1, "initial_score": float(scores[3, 1]), "label_loss": .8},
                {"key_block": 2, "initial_score": float(scores[3, 2]), "label_loss": .9},
            ]
            (path / "experiment.json").write_text(json.dumps(meta), encoding="utf-8")
            (path / "trials.jsonl").write_text(
                "\n".join(json.dumps(row) for row in trials) + "\n", encoding="utf-8")
            np.savez(path / "block_map.npz", initial_scores=scores, selected_mask=mask)
            run = load_run(path, kernel_size=3)
            self.assertEqual(run["layer"], 2)
            np.testing.assert_allclose(run["utility"], [.2, .1])
            self.assertEqual(run["patches"].shape, (2, 3, 3))
            (path / "trials.jsonl").write_text(json.dumps(trials[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every candidate"):
                load_run(path, kernel_size=3)

    def test_command_entry_point_writes_complete_artifact_set(self):
        from run_observation3 import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "swap", root / "observation3"
            source.mkdir()
            rng = np.random.default_rng(12)
            size, q = 110, 100
            scores = rng.normal(size=(size, size))
            keys = np.arange(5, 95)
            patches = np.asarray([extract_replicate_patch(scores, q, int(k), 7) for k in keys])
            utility = .01 * (patches[:, 2, 5] - .8 * patches[:, 4, 1])
            mask = np.zeros_like(scores, dtype=bool)
            mask[q, q] = True
            meta = {
                "status": "complete",
                "arguments": {"layer": 4, "head": 2, "sample_index": 0,
                              "data": "same_prompt.jsonl"},
                "plan": {"query_block": q, "removed_key_block": q,
                         "candidate_key_blocks": keys.tolist()},
                "baseline": {"label_loss": 5.0},
            }
            trials = [{"key_block": int(key), "initial_score": float(scores[q, key]),
                       "label_loss": float(5.0 - utility[index])}
                      for index, key in enumerate(keys)]
            (source / "experiment.json").write_text(json.dumps(meta), encoding="utf-8")
            (source / "trials.jsonl").write_text(
                "\n".join(json.dumps(row) for row in trials) + "\n", encoding="utf-8")
            np.savez(source / "block_map.npz", initial_scores=scores, selected_mask=mask)
            argv = ["run_observation3.py", "--input", str(source), "--output", str(output),
                    "--permutations", "100", "--bootstrap-samples", "100", "--seed", "7"]
            with mock.patch("sys.argv", argv):
                self.assertEqual(main(), 0)
            for filename in ("summary.json", "observation3.png", "observation3.pdf",
                             "per_run_metrics.csv", "oof_predictions.csv",
                             "candidate_patches.csv", "kernel_weights.csv",
                             "offset_correlations.csv", "ablation_metrics.csv",
                             "alpha_sensitivity.csv"):
                self.assertTrue((output / filename).is_file(), filename)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete")
            self.assertNotIn("NaN", (output / "summary.json").read_text(encoding="utf-8"))

    @staticmethod
    def synthetic_run(seed=9):
        rng = np.random.default_rng(seed)
        size, q = 220, 190
        scores = rng.normal(size=(size, size))
        keys = np.arange(8, 181)
        patches = np.asarray([extract_replicate_patch(scores, q, int(k), 7) for k in keys])
        utility = (1.4 * patches[:, 2, 5] - 1.1 * patches[:, 4, 1]
                   + .2 * patches[:, 3, 3] + rng.normal(scale=.03, size=len(keys)))
        return {
            "run_id": "synthetic", "source": "synthetic", "meta": {"arguments": {}},
            "layer": 0, "head": 0, "query_block": q, "removed_key_block": q,
            "baseline_loss": 1.0, "keys": keys, "center": patches[:, 3, 3],
            "patches": patches, "replacement_loss": 1.0 - utility,
            "utility": utility, "kernel_size": 7,
        }


if __name__ == "__main__":
    unittest.main()
