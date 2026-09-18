"""Pure NumPy analysis for Observation 2: local score context and block utility."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def causal_neighborhood(scores, query_block, key_block, line_radius=3):
    """Return vertical and ``\\``-diagonal neighbors around a candidate.

    ``line_radius=3`` gives six positions on each line: three on either
    side of the center.  The vertical line varies the query-block index while
    keeping the key-block index fixed.  The ``\\`` diagonal varies both
    indices by the same signed offset.  The center is excluded, and only
    in-bounds causally valid positions are retained.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("initial_scores must be a square 2-D block map")
    if line_radius < 1:
        raise ValueError("line_radius must be a positive integer")
    q, k = int(query_block), int(key_block)
    if not (0 <= q < scores.shape[0] and 0 <= k < scores.shape[1] and k <= q):
        raise ValueError("center block must be in bounds and causally valid")
    values = []
    coordinates = []
    for offset in range(1, line_radius + 1):
        for nq, nk in ((q - offset, k), (q + offset, k),
                       (q - offset, k - offset), (q + offset, k + offset)):
            if (0 <= nq < scores.shape[0] and 0 <= nk < scores.shape[1]
                    and nk <= nq and (nq, nk) != (q, k)):
                value = float(scores[nq, nk])
                if not math.isfinite(value):
                    raise ValueError("Non-finite causally valid score in neighborhood")
                values.append(value)
                coordinates.append((nq, nk))
    if not values:
        raise ValueError("No valid neighbors around candidate")
    return np.asarray(values, dtype=np.float64), coordinates


def load_run(directory, line_radius=3, run_id=None):
    """Load one complete block-swap sweep and derive outcome-blind context features."""
    directory = Path(directory).resolve()
    meta_file = directory / "experiment.json"
    trials_file = directory / "trials.jsonl"
    map_file = directory / "block_map.npz"
    for filename in (meta_file, trials_file, map_file):
        if not filename.is_file():
            raise FileNotFoundError(f"Missing first-observation result: {filename}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError(f"Refusing incomplete sweep: {directory}")
    trials = [json.loads(line) for line in trials_file.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    arrays = np.load(map_file)
    scores = np.asarray(arrays["initial_scores"], dtype=np.float64)
    mask = np.asarray(arrays["selected_mask"], dtype=bool)
    if scores.shape != mask.shape or scores.ndim != 2:
        raise ValueError("initial_scores and selected_mask must be same-shaped 2-D arrays")
    plan = meta["plan"]
    q = int(plan["query_block"])
    removed = int(plan["removed_key_block"])
    expected = [int(k) for k in plan["candidate_key_blocks"]]
    keys = [int(row["key_block"]) for row in trials]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("trials.jsonl must contain every candidate exactly once")
    if not mask[q, removed] or any(mask[q, key] for key in expected):
        raise ValueError("Saved baseline mask disagrees with intervention plan")
    baseline = float(meta["baseline"]["label_loss"])
    tolerance = float(meta.get("summary", {}).get("effective_improvement_tolerance", 0.0))
    if not math.isfinite(baseline) or tolerance < 0:
        raise ValueError("Invalid baseline loss or numerical tolerance")
    rid = str(run_id if run_id is not None else directory.name)
    records = []
    by_key = {int(row["key_block"]): row for row in trials}
    for key in expected:
        row = by_key[key]
        center = float(scores[q, key])
        logged_center = float(row["initial_score"])
        loss = float(row["label_loss"])
        if not (math.isfinite(center) and math.isfinite(loss)):
            raise ValueError("Non-finite center score or label loss")
        if not np.isclose(center, logged_center, rtol=1e-5, atol=1e-8):
            raise ValueError(f"Logged score differs from block map for key block {key}")
        neighbors, coordinates = causal_neighborhood(scores, q, key, line_radius)
        records.append({
            "run_id": rid,
            "source": str(directory),
            "query_block": q,
            "key_block": key,
            "center_score": center,
            "neighbor_mean": float(neighbors.mean()),
            "neighbor_std": float(neighbors.std()),
            "neighbor_count": int(len(neighbors)),
            "replacement_loss": loss,
            "baseline_loss": baseline,
            "utility": baseline - loss,
            "delta_loss": loss - baseline,
            "effective_tolerance": tolerance,
            "improves_baseline": bool(baseline - loss > tolerance),
            "neighbor_coordinates": coordinates,
        })
    return {"directory": str(directory), "run_id": rid, "meta": meta,
            "scores": scores, "mask": mask, "records": records}


def _zscore(values):
    values = np.asarray(values, dtype=np.float64)
    scale = values.std()
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("Cannot standardize a constant/non-finite feature")
    return (values - values.mean()) / scale


def match_by_center(records, score_caliper=0.15):
    """Match high/low-neighborhood candidates using center score only.

    Grouping and pairing never inspect label loss or utility.
    The caliper is measured in within-run center-score standard deviations.
    """
    if score_caliper <= 0:
        raise ValueError("score_caliper must be positive")
    records = list(records)
    if len(records) < 4:
        raise ValueError("At least four candidates are required")
    center_z = _zscore([r["center_score"] for r in records])
    neighborhood = np.asarray([r["neighbor_mean"] for r in records])
    median = float(np.median(neighborhood))
    low = [i for i, value in enumerate(neighborhood) if value < median]
    high = [i for i, value in enumerate(neighborhood) if value >= median]
    if not low or not high:
        raise ValueError("Neighborhood split produced an empty group")
    possible = []
    for li in low:
        for hi in high:
            gap = abs(float(center_z[li] - center_z[hi]))
            if gap <= score_caliper:
                # Secondary key maximizes context separation but remains outcome-blind.
                context_gap = float(neighborhood[hi] - neighborhood[li])
                possible.append((gap, -context_gap, records[li]["key_block"],
                                 records[hi]["key_block"], li, hi))
    possible.sort()
    used_low, used_high, pairs = set(), set(), []
    for gap, _, _, _, li, hi in possible:
        if li in used_low or hi in used_high:
            continue
        used_low.add(li)
        used_high.add(hi)
        lo, high_record = records[li], records[hi]
        utility_diff = float(high_record["utility"] - lo["utility"])
        tolerance = max(float(lo["effective_tolerance"]),
                        float(high_record["effective_tolerance"]))
        pairs.append({
            "run_id": lo["run_id"],
            "low_key_block": int(lo["key_block"]),
            "high_key_block": int(high_record["key_block"]),
            "low_center_score": float(lo["center_score"]),
            "high_center_score": float(high_record["center_score"]),
            "center_z_gap": float(gap),
            "low_neighbor_mean": float(lo["neighbor_mean"]),
            "high_neighbor_mean": float(high_record["neighbor_mean"]),
            "neighbor_gap": float(high_record["neighbor_mean"] - lo["neighbor_mean"]),
            "low_loss": float(lo["replacement_loss"]),
            "high_loss": float(high_record["replacement_loss"]),
            "low_utility": float(lo["utility"]),
            "high_utility": float(high_record["utility"]),
            "utility_difference": utility_diff,
            "effective_tolerance": tolerance,
            "outcome": "high_context_better" if utility_diff > tolerance else
                       "low_context_better" if utility_diff < -tolerance else "tie",
        })
    return pairs


def residualize_by_center(records):
    """Remove an intercept and linear center-score effect within each run."""
    records = list(records)
    output = []
    run_ids = sorted(set(r["run_id"] for r in records))
    for run_id in run_ids:
        subset = [r for r in records if r["run_id"] == run_id]
        if len(subset) < 3:
            raise ValueError("Each run needs at least three candidates")
        center = _zscore([r["center_score"] for r in subset])
        context = _zscore([r["neighbor_mean"] for r in subset])
        utility = np.asarray([r["utility"] for r in subset], dtype=np.float64)
        design = np.column_stack([np.ones(len(subset)), center])
        context_residual = context - design @ np.linalg.lstsq(design, context, rcond=None)[0]
        utility_residual = utility - design @ np.linalg.lstsq(design, utility, rcond=None)[0]
        for record, c_res, u_res in zip(subset, context_residual, utility_residual):
            output.append({**record, "context_residual": float(c_res),
                           "utility_residual": float(u_res)})
    return output


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("Correlation inputs must be same-length vectors")
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def rankdata(values):
    """Average ranks for ties, implemented without SciPy."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def monte_carlo_sign_flip(differences, permutations=20000, seed=42):
    differences = np.asarray(differences, dtype=np.float64)
    if not len(differences) or permutations < 100:
        raise ValueError("Need paired differences and at least 100 permutations")
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    count_greater, count_abs = 0, 0
    remaining = permutations
    while remaining:
        batch = min(2000, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(differences)))
        statistics = (signs * differences).mean(axis=1)
        count_greater += int(np.count_nonzero(statistics >= observed))
        count_abs += int(np.count_nonzero(np.abs(statistics) >= abs(observed)))
        remaining -= batch
    return {"method": "Monte Carlo paired sign-flip", "permutations": int(permutations),
            "observed_mean": observed,
            "one_sided_p_high_context_better": (count_greater + 1) / (permutations + 1),
            "two_sided_p": (count_abs + 1) / (permutations + 1)}


def paired_bootstrap_ci(differences, samples=20000, seed=43):
    differences = np.asarray(differences, dtype=np.float64)
    if not len(differences) or samples < 100:
        raise ValueError("Need paired differences and at least 100 bootstrap samples")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 2000):
        count = min(2000, samples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        means[start:start + count] = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def residual_permutation(records, permutations=20000, seed=44):
    """Permutation association test, shuffling utility residuals within each run."""
    records = list(records)
    x = np.asarray([r["context_residual"] for r in records], dtype=np.float64)
    y = np.asarray([r["utility_residual"] for r in records], dtype=np.float64)
    observed = pearson(x, y)
    groups = [[i for i, r in enumerate(records) if r["run_id"] == run_id]
              for run_id in sorted(set(r["run_id"] for r in records))]
    rng = np.random.default_rng(seed)
    greater = absolute = 0
    for _ in range(permutations):
        shuffled = y.copy()
        for indices in groups:
            shuffled[indices] = rng.permutation(shuffled[indices])
        statistic = pearson(x, shuffled)
        greater += statistic >= observed
        absolute += abs(statistic) >= abs(observed)
    return {"method": "Within-run residual permutation", "permutations": int(permutations),
            "partial_pearson_r": observed,
            "one_sided_p_positive": (greater + 1) / (permutations + 1),
            "two_sided_p": (absolute + 1) / (permutations + 1)}


def summarize(records, pairs, permutations=20000, bootstrap_samples=20000, seed=42):
    records, pairs = list(records), list(pairs)
    if not pairs:
        raise ValueError("No matched pairs; increase --score-caliper or add runs")
    differences = np.asarray([p["utility_difference"] for p in pairs])
    residual = residualize_by_center(records)
    context_r = pearson([r["context_residual"] for r in residual],
                        [r["utility_residual"] for r in residual])
    context_spearman = pearson(rankdata([r["context_residual"] for r in residual]),
                               rankdata([r["utility_residual"] for r in residual]))
    test = monte_carlo_sign_flip(differences, permutations, seed)
    residual_test = residual_permutation(residual, permutations, seed + 2)
    wins = sum(p["outcome"] == "high_context_better" for p in pairs)
    losses = sum(p["outcome"] == "low_context_better" for p in pairs)
    ties = len(pairs) - wins - losses
    return {
        "observation": "Neighborhood scores contain block-utility information beyond the center score.",
        "utility_definition": "baseline label NLL minus replacement label NLL; higher is better",
        "neighborhood_definition": "mean initial score of distinct in-bounds causally-valid blocks on the vertical and backslash-diagonal lines, three blocks on each side per line, center excluded",
        "candidate_count": len(records),
        "run_count": len(set(r["run_id"] for r in records)),
        "matched_pair_count": len(pairs),
        "mean_center_score_gap_in_sd": float(np.mean([p["center_z_gap"] for p in pairs])),
        "max_center_score_gap_in_sd": float(np.max([p["center_z_gap"] for p in pairs])),
        "mean_neighbor_score_gap": float(np.mean([p["neighbor_gap"] for p in pairs])),
        "mean_utility_difference_high_minus_low_context": float(differences.mean()),
        "paired_bootstrap_95pct_ci": paired_bootstrap_ci(differences, bootstrap_samples, seed + 1),
        "high_context_wins": wins,
        "low_context_wins": losses,
        "ties_within_numerical_tolerance": ties,
        "high_context_win_fraction_excluding_ties": wins / (wins + losses) if wins + losses else None,
        "paired_sign_flip_test": test,
        "partial_neighbor_pearson_r": context_r,
        "partial_neighbor_spearman_r": context_spearman,
        "residual_permutation_test": residual_test,
        "supports_observation_in_this_dataset": bool(
            differences.mean() > 0 and test["one_sided_p_high_context_better"] < 0.05
            and context_r > 0 and residual_test["one_sided_p_positive"] < 0.05
        ),
        "scope_warning": "A single run is an illustrative within-row study; aggregate prespecified samples/layers/heads before making a general claim.",
    }, residual
