"""Experiment planning and summaries. No model/GPU dependencies."""
from __future__ import annotations

import math
import numpy as np


def make_plan(initial_scores, selected_mask, query_block):
    scores = np.asarray(initial_scores, dtype=np.float64)
    mask = np.asarray(selected_mask, dtype=bool)
    if scores.ndim != 2 or scores.shape != mask.shape:
        raise ValueError("Expected same-shaped [query_blocks, key_blocks] arrays")
    if not 0 <= query_block < scores.shape[0]:
        raise ValueError("query_block is outside the prompt")
    if np.any(mask & ~np.tri(*mask.shape, dtype=bool)):
        raise ValueError("Baseline mask contains future blocks")
    valid = np.arange(scores.shape[1]) <= query_block
    if not np.isfinite(scores[query_block, valid]).all():
        raise ValueError("Non-finite initial scores")
    selected = np.flatnonzero(mask[query_block] & valid)
    candidates = np.flatnonzero(~mask[query_block] & valid)
    if not len(selected) or not len(candidates):
        raise ValueError("Choose a query row with both selected and unselected legal blocks")
    # Stable tie break: smallest key-block index; do not optimize by label loss.
    removed = int(selected[np.argmin(scores[query_block, selected])])
    return {
        "query_block": int(query_block),
        "removed_key_block": removed,
        "removed_initial_score": float(scores[query_block, removed]),
        "selected_key_blocks": selected.tolist(),
        "candidate_key_blocks": candidates.tolist(),
    }


def swapped_mask(mask, plan, candidate):
    if candidate not in plan["candidate_key_blocks"]:
        raise ValueError("Replacement must be unselected and causally valid")
    before = np.asarray(mask, dtype=bool)
    after = before.copy()
    q, removed = plan["query_block"], plan["removed_key_block"]
    after[q, removed], after[q, candidate] = False, True
    if np.count_nonzero(before != after) != 2:
        raise AssertionError("A trial must change exactly two mask entries")
    if not np.array_equal(before.sum(axis=1), after.sum(axis=1)):
        raise AssertionError("Per-query block budget changed")
    return after


def summarize(baseline, repeated, rows, removed_score, tolerance):
    if not rows or not all(math.isfinite(float(r["label_loss"])) for r in rows):
        raise ValueError("A complete finite candidate sweep is required")
    drift = abs(float(baseline) - float(repeated))
    effective = max(float(tolerance), 5.0 * drift)
    lower = [r for r in rows if r["initial_score"] < removed_score]
    improved = [r for r in lower if baseline - r["label_loss"] > effective]
    return {
        "baseline_label_loss": float(baseline),
        "repeated_baseline_label_loss": float(repeated),
        "baseline_repeat_drift": drift,
        "effective_improvement_tolerance": effective,
        "candidate_count": len(rows),
        "strictly_lower_score_candidates": len(lower),
        "lower_score_better_loss_count": len(improved),
        "lower_score_better_loss_fraction": len(improved) / len(lower) if lower else None,
        "best_candidate_key_block": min(rows, key=lambda r: r["label_loss"])["key_block"],
        "has_counterexample": bool(improved),
        "interpretation": (
            "An empirical counterexample exists for this sample/layer/head/query row."
            if improved else
            "This sweep does not establish the proposed observation."
        ),
    }
