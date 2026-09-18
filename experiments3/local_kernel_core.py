"""Observation 3: signed 7x7 local score patterns versus block utility.

The analysis is NumPy-only.  Patch extraction intentionally mirrors
``xattn.src.Conv.apply_conv2d_block_map``: odd square kernels, PyTorch-style
cross-correlation orientation, and replicate padding at map boundaries.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def extract_replicate_patch(scores, query_block, key_block, kernel_size=7):
    """Extract the exact local patch consumed by Conv.py at one map position."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("initial_scores must be a square 2-D block map")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    q, k = int(query_block), int(key_block)
    if not (0 <= q < scores.shape[0] and 0 <= k < scores.shape[1] and k <= q):
        raise ValueError("center block must be in bounds and causally valid")
    if not np.isfinite(scores).all():
        raise ValueError("initial_scores contains NaN or Inf")
    radius = kernel_size // 2
    padded = np.pad(scores, ((radius, radius), (radius, radius)), mode="edge")
    patch = padded[q:q + kernel_size, k:k + kernel_size].copy()
    if patch.shape != (kernel_size, kernel_size):
        raise AssertionError("replicate-padded patch has the wrong shape")
    if not np.isclose(patch[radius, radius], scores[q, k], rtol=0, atol=0):
        raise AssertionError("patch center differs from candidate score")
    return patch


def load_run(directory, kernel_size=7, run_id=None):
    """Load one complete Observation-1 swap sweep and construct 7x7 patches."""
    directory = Path(directory).resolve()
    meta_file = directory / "experiment.json"
    trials_file = directory / "trials.jsonl"
    map_file = directory / "block_map.npz"
    for filename in (meta_file, trials_file, map_file):
        if not filename.is_file():
            raise FileNotFoundError(f"Missing Observation-1 result: {filename}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError(f"Refusing incomplete swap sweep: {directory}")
    trials = [json.loads(line) for line in trials_file.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    arrays = np.load(map_file)
    scores = np.asarray(arrays["initial_scores"], dtype=np.float64)
    mask = np.asarray(arrays["selected_mask"], dtype=bool)
    if scores.shape != mask.shape or scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("initial_scores and selected_mask must be same-shaped square maps")
    plan = meta["plan"]
    q = int(plan["query_block"])
    removed = int(plan["removed_key_block"])
    expected = [int(k) for k in plan["candidate_key_blocks"]]
    keys = [int(row["key_block"]) for row in trials]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("trials.jsonl must contain every candidate exactly once")
    if not mask[q, removed] or any(mask[q, key] for key in expected):
        raise ValueError("Saved baseline mask disagrees with the intervention plan")
    baseline = float(meta["baseline"]["label_loss"])
    if not math.isfinite(baseline):
        raise ValueError("Baseline label loss is not finite")
    by_key = {int(row["key_block"]): row for row in trials}
    ordered = sorted(expected)
    patches, centers, losses, utilities = [], [], [], []
    for key in ordered:
        row = by_key[key]
        center = float(scores[q, key])
        logged_center = float(row["initial_score"])
        loss = float(row["label_loss"])
        if not (math.isfinite(center) and math.isfinite(loss)):
            raise ValueError("Non-finite center score or replacement loss")
        if not np.isclose(center, logged_center, rtol=1e-5, atol=1e-8):
            raise ValueError(f"Logged score differs from block map for key block {key}")
        patches.append(extract_replicate_patch(scores, q, key, kernel_size))
        centers.append(center)
        losses.append(loss)
        utilities.append(baseline - loss)
    args = meta.get("arguments", {})
    rid = str(run_id if run_id is not None else directory.name)
    return {
        "run_id": rid,
        "source": str(directory),
        "meta": meta,
        "layer": int(args.get("layer", -1)),
        "head": int(args.get("head", -1)),
        "query_block": q,
        "removed_key_block": removed,
        "baseline_loss": baseline,
        "keys": np.asarray(ordered, dtype=np.int64),
        "center": np.asarray(centers, dtype=np.float64),
        "patches": np.asarray(patches, dtype=np.float64),
        "replacement_loss": np.asarray(losses, dtype=np.float64),
        "utility": np.asarray(utilities, dtype=np.float64),
        "kernel_size": int(kernel_size),
    }


def rankdata(values):
    """Average ranks for ties, without SciPy."""
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


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("Correlation vectors must have equal one-dimensional shape")
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def feature_indices(kernel_size=7, subset="full"):
    """Return flattened off-center patch positions for one ablation."""
    radius = kernel_size // 2
    result = []
    for row in range(kernel_size):
        for col in range(kernel_size):
            dq, dk = row - radius, col - radius
            if dq == 0 and dk == 0:
                continue
            keep = (
                subset == "full"
                or (subset == "vertical_diag" and (dk == 0 or dq == dk))
                or (subset == "same_row" and dq == 0)
                or (subset == "same_column" and dk == 0)
                or (subset == "diagonals" and abs(dq) == abs(dk))
            )
            if keep:
                result.append(row * kernel_size + col)
    if subset not in {"full", "vertical_diag", "same_row", "same_column", "diagonals"}:
        raise ValueError(f"Unknown feature subset: {subset}")
    return np.asarray(result, dtype=np.int64)


def blocked_purged_folds(keys, folds=5, purge_radius=6):
    """Contiguous key-block test folds with overlapping train patches removed."""
    keys = np.asarray(keys, dtype=np.int64)
    if folds < 2 or folds > len(keys):
        raise ValueError("folds must be between 2 and candidate count")
    if purge_radius < 0:
        raise ValueError("purge_radius must be nonnegative")
    order = np.argsort(keys, kind="mergesort")
    output = []
    for chunk in np.array_split(order, folds):
        test = np.asarray(chunk, dtype=np.int64)
        if not len(test):
            continue
        train_mask = np.ones(len(keys), dtype=bool)
        train_mask[test] = False
        distance = np.min(np.abs(keys[:, None] - keys[test][None, :]), axis=1)
        train_mask &= distance > purge_radius
        train = np.flatnonzero(train_mask)
        if len(train) < 4:
            raise ValueError("Purging leaves too few training candidates")
        output.append((train, test))
    covered = np.concatenate([test for _, test in output])
    if len(covered) != len(keys) or len(np.unique(covered)) != len(keys):
        raise AssertionError("Blocked folds do not cover every candidate exactly once")
    return output


def _safe_scale(values):
    scale = float(np.std(values))
    return scale if math.isfinite(scale) and scale > 1e-12 else 1.0


def _control_design(run, train, mode):
    """Build train-fitted control columns for every candidate."""
    center = np.asarray(run["center"], dtype=np.float64)
    keys = np.asarray(run["keys"], dtype=np.float64)
    center_mean, center_scale = float(center[train].mean()), _safe_scale(center[train])
    center_z = (center - center_mean) / center_scale
    columns = [np.ones(len(center)), center_z]
    metadata = {"center_mean": center_mean, "center_scale": center_scale}
    if mode == "center_position":
        key_mean, key_scale = float(keys[train].mean()), _safe_scale(keys[train])
        key_z = (keys - key_mean) / key_scale
        columns.extend([key_z, key_z ** 2])
        metadata.update(key_mean=key_mean, key_scale=key_scale)
    elif mode != "center":
        raise ValueError("control mode must be center or center_position")
    return np.column_stack(columns), metadata


def prepare_cv_plans(run, folds=5, purge_radius=6, ridge_alpha=0.1,
                     control_mode="center_position", subset="full"):
    """Precompute feature-only transforms and linear operators for CV."""
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be nonnegative")
    kernel_size = int(run["kernel_size"])
    indices = feature_indices(kernel_size, subset)
    flat = np.asarray(run["patches"], dtype=np.float64).reshape(len(run["keys"]), -1)
    x_all = flat[:, indices]
    plans = []
    for fold_index, (train, test) in enumerate(
            blocked_purged_folds(run["keys"], folds, purge_radius)):
        controls, control_meta = _control_design(run, train, control_mode)
        b_train, b_test = controls[train], controls[test]
        b_pinv = np.linalg.pinv(b_train)
        feature_on_controls = b_pinv @ x_all[train]
        residual_train = x_all[train] - b_train @ feature_on_controls
        residual_test = x_all[test] - b_test @ feature_on_controls
        scale = residual_train.std(axis=0)
        active = np.isfinite(scale) & (scale > 1e-12)
        z_train = residual_train[:, active] / scale[active]
        z_test = residual_test[:, active] / scale[active]
        if z_train.shape[1]:
            gram = (z_train.T @ z_train) / len(train)
            rhs = z_train.T / len(train)
            ridge_operator = np.linalg.solve(
                gram + float(ridge_alpha) * np.eye(z_train.shape[1]), rhs)
        else:
            ridge_operator = np.zeros((0, len(train)), dtype=np.float64)
        plans.append({
            "fold": fold_index,
            "train": train,
            "test": test,
            "b_train": b_train,
            "b_test": b_test,
            "b_pinv": b_pinv,
            "feature_on_controls": feature_on_controls,
            "scale": scale,
            "active": active,
            "z_train": z_train,
            "z_test": z_test,
            "ridge_operator": ridge_operator,
            "feature_indices": indices,
            "control_meta": control_meta,
            "control_mode": control_mode,
            "subset": subset,
        })
    return plans


def evaluate_cv(run, plans, utility=None, return_kernels=False):
    """Fit center/control baseline and signed local ridge model out of sample."""
    y = np.asarray(run["utility"] if utility is None else utility, dtype=np.float64)
    n = len(y)
    base_prediction = np.full(n, np.nan, dtype=np.float64)
    full_prediction = np.full(n, np.nan, dtype=np.float64)
    kernels = []
    kernel_size = int(run["kernel_size"])
    radius = kernel_size // 2
    for plan in plans:
        train, test = plan["train"], plan["test"]
        beta = plan["b_pinv"] @ y[train]
        pred_base = plan["b_test"] @ beta
        train_residual = y[train] - plan["b_train"] @ beta
        theta = plan["ridge_operator"] @ train_residual
        pred_full = pred_base + plan["z_test"] @ theta
        base_prediction[test] = pred_base
        full_prediction[test] = pred_full
        if return_kernels:
            raw_feature_weight = np.zeros(len(plan["feature_indices"]), dtype=np.float64)
            raw_feature_weight[plan["active"]] = theta / plan["scale"][plan["active"]]
            adjusted_controls = beta - plan["feature_on_controls"] @ raw_feature_weight
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
            kernel.flat[plan["feature_indices"]] = raw_feature_weight
            center_weight = adjusted_controls[1] / plan["control_meta"]["center_scale"]
            kernel[radius, radius] += center_weight
            norm = float(np.linalg.norm(kernel))
            kernels.append({
                "fold": int(plan["fold"]),
                "kernel": kernel,
                "normalized_kernel": kernel / norm if norm > 0 else kernel.copy(),
                "positive_count": int(np.count_nonzero(kernel > 1e-12)),
                "negative_count": int(np.count_nonzero(kernel < -1e-12)),
                "l2_norm": norm,
            })
    valid = np.isfinite(base_prediction) & np.isfinite(full_prediction) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        raise ValueError("Too few out-of-fold predictions")
    yv, bv, fv = y[valid], base_prediction[valid], full_prediction[valid]
    mse_base = float(np.mean((yv - bv) ** 2))
    mse_full = float(np.mean((yv - fv) ** 2))
    variance = float(np.mean((yv - yv.mean()) ** 2))
    relative = ((mse_base - mse_full) / mse_base
                if mse_base > 1e-24 else float("nan"))
    metrics = {
        "candidate_count": int(len(y)),
        "oof_count": int(np.count_nonzero(valid)),
        "mse_control_baseline": mse_base,
        "mse_signed_local_kernel": mse_full,
        "mse_reduction": float(mse_base - mse_full),
        "relative_mse_reduction": float(relative),
        "incremental_oof_r2": (float((mse_base - mse_full) / variance)
                               if variance > 1e-24 else float("nan")),
        "r2_control_baseline": float(1 - mse_base / variance) if variance > 0 else float("nan"),
        "r2_signed_local_kernel": float(1 - mse_full / variance) if variance > 0 else float("nan"),
        "prediction_pearson_r": pearson(yv, fv),
        "prediction_spearman_r": pearson(rankdata(yv), rankdata(fv)),
        "utility_std": float(np.std(yv)),
    }
    return {
        "metrics": metrics,
        "base_prediction": base_prediction,
        "full_prediction": full_prediction,
        "valid": valid,
        "kernels": kernels,
    }


def prepare_offset_state(run, control_mode="center_position"):
    """Residualize every off-center patch cell against run-level controls."""
    n = len(run["keys"])
    all_indices = feature_indices(run["kernel_size"], "full")
    flat = run["patches"].reshape(n, -1)
    controls, _ = _control_design(run, np.arange(n), control_mode)
    controls_pinv = np.linalg.pinv(controls)
    x_residual = flat[:, all_indices] - controls @ (controls_pinv @ flat[:, all_indices])
    x_scale = x_residual.std(axis=0)
    active = np.isfinite(x_scale) & (x_scale > 1e-12)
    x_z = np.zeros_like(x_residual)
    x_z[:, active] = x_residual[:, active] / x_scale[active]
    return {
        "controls": controls,
        "controls_pinv": controls_pinv,
        "x_z": x_z,
        "active": active,
        "feature_indices": all_indices,
    }


def pooled_offset_correlations(runs, states, utilities=None):
    """Signed partial correlation at each off-center kernel position."""
    kernel_size = int(runs[0]["kernel_size"])
    numerator = np.zeros(kernel_size * kernel_size, dtype=np.float64)
    x_ss = np.zeros_like(numerator)
    y_ss = 0.0
    for index, (run, state) in enumerate(zip(runs, states)):
        y = np.asarray(run["utility"] if utilities is None else utilities[index], dtype=np.float64)
        y_residual = y - state["controls"] @ (state["controls_pinv"] @ y)
        y_scale = float(np.std(y_residual))
        if not math.isfinite(y_scale) or y_scale <= 1e-12:
            continue
        y_z = y_residual / y_scale
        x_z = state["x_z"]
        local_num = x_z.T @ y_z
        local_xss = np.sum(x_z ** 2, axis=0)
        numerator[state["feature_indices"]] += local_num
        x_ss[state["feature_indices"]] += local_xss
        y_ss += float(np.dot(y_z, y_z))
    denominator = np.sqrt(x_ss * y_ss)
    corr = np.full(kernel_size * kernel_size, np.nan, dtype=np.float64)
    valid = denominator > 0
    corr[valid] = numerator[valid] / denominator[valid]
    return corr.reshape(kernel_size, kernel_size)


def dependency_group(run):
    """Identify runs that reuse one prompt/sample and must be permuted together."""
    arguments = run.get("meta", {}).get("arguments", {})
    data_source = str(arguments.get("data", arguments.get("input", "")))
    sample_index = str(arguments.get("sample_index", ""))
    return (data_source, sample_index, int(run["query_block"]),
            tuple(int(x) for x in run["keys"]))


def aggregate_incremental_r2_effect(results, runs=None):
    """Dependency-group-balanced mean of within-run incremental OOF R-squared."""
    raw = [float(result["metrics"]["incremental_oof_r2"]) for result in results]
    values = np.asarray([value for value in raw if math.isfinite(value)], dtype=np.float64)
    if runs is None:
        aggregate_values = values
    else:
        grouped = {}
        for run, value in zip(runs, raw):
            if math.isfinite(value):
                grouped.setdefault(dependency_group(run), []).append(value)
        aggregate_values = np.asarray([np.mean(group) for group in grouped.values()],
                                      dtype=np.float64)
    mean = float(aggregate_values.mean()) if len(aggregate_values) else float("nan")
    return mean, values


def permutation_tests(runs, plans_by_run, offset_states, observed_effect,
                      observed_correlations, permutations=2000, seed=42):
    """Circular-shift null tests preserving within-run utility structure."""
    if permutations < 100:
        raise ValueError("Use at least 100 permutations")
    rng = np.random.default_rng(seed)
    observed_abs = np.abs(observed_correlations.ravel())
    observed_max = float(np.nanmax(observed_abs))
    effect_ge = 0
    max_ge = 0
    offset_ge = np.zeros_like(observed_abs, dtype=np.int64)
    groups = {}
    for run_index, run in enumerate(runs):
        groups.setdefault(dependency_group(run), []).append(run_index)
    group_members = list(groups.values())
    if len(group_members) == 1:
        length = len(runs[group_members[0][0]]["utility"])
        shift_schedule = [(shift,) for shift in range(1, length)]
        method = "exact synchronized within-run circular-shift permutation"
    else:
        shift_schedule = [tuple(int(rng.integers(1, len(runs[members[0]]["utility"])))
                                for members in group_members)
                          for _ in range(permutations)]
        method = "Monte Carlo dependency-synchronized within-run circular-shift permutation"
    evaluated = len(shift_schedule)
    null_effects = np.empty(evaluated, dtype=np.float64)
    null_max = np.empty(evaluated, dtype=np.float64)
    for iteration, shifts in enumerate(shift_schedule):
        permuted = [None] * len(runs)
        for member_indices, shift in zip(group_members, shifts):
            length = len(runs[member_indices[0]]["utility"])
            if any(len(runs[index]["utility"]) != length for index in member_indices):
                raise AssertionError("Dependency-group runs have incompatible candidate counts")
            for index in member_indices:
                permuted[index] = np.roll(runs[index]["utility"], shift)
        perm_results = [evaluate_cv(run, plans, utility=y, return_kernels=False)
                        for run, plans, y in zip(runs, plans_by_run, permuted)]
        statistic, _ = aggregate_incremental_r2_effect(perm_results, runs)
        corr = pooled_offset_correlations(runs, offset_states, permuted)
        maximum = float(np.nanmax(np.abs(corr)))
        null_effects[iteration] = statistic
        null_max[iteration] = maximum
        effect_ge += statistic >= observed_effect
        max_ge += maximum >= observed_max
        offset_ge += maximum >= observed_abs
    corrected = (offset_ge + 1) / (evaluated + 1)
    return {
        "method": method,
        "dependency_handling": ("Runs that reuse the same data/sample, query block, and candidate "
                                "grid receive the same circular shift."),
        "dependency_group_count": int(len(groups)),
        "requested_permutations": int(permutations),
        "evaluated_null_shifts": int(evaluated),
        "cv_effect_one_sided_p": (effect_ge + 1) / (evaluated + 1),
        "any_offset_max_abs_p": (max_ge + 1) / (evaluated + 1),
        "offset_fwer_p": corrected.reshape(observed_correlations.shape),
        "null_cv_effect_quantiles": [float(x) for x in np.percentile(null_effects, [2.5, 50, 97.5])],
        "null_max_abs_correlation_quantiles": [float(x) for x in np.percentile(null_max, [2.5, 50, 97.5])],
    }


def bootstrap_mean_ci(values, samples=10000, seed=43):
    values = np.asarray([x for x in values if math.isfinite(float(x))], dtype=np.float64)
    if not len(values):
        return [float("nan"), float("nan")]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 2000):
        count = min(2000, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def summarize_analysis(runs, results, correlations, permutation, ridge_alpha,
                       folds, purge_radius, control_mode, bootstrap_samples=10000,
                       seed=42):
    mean_effect, effects = aggregate_incremental_r2_effect(results, runs)
    flat = correlations.ravel()
    finite_indices = np.flatnonzero(np.isfinite(flat))
    best_flat = int(finite_indices[np.argmax(np.abs(flat[finite_indices]))])
    kernel_size = correlations.shape[0]
    radius = kernel_size // 2
    best_row, best_col = divmod(best_flat, kernel_size)
    max_corr = float(flat[best_flat])
    p_cv = float(permutation["cv_effect_one_sided_p"])
    p_offset = float(permutation["any_offset_max_abs_p"])
    grouped_effects = {}
    for run, result in zip(runs, results):
        value = float(result["metrics"]["incremental_oof_r2"])
        if math.isfinite(value):
            grouped_effects.setdefault(dependency_group(run), []).append(value)
    group_means = np.asarray([np.mean(values) for values in grouped_effects.values()],
                             dtype=np.float64)
    # The prespecified primary claim is collective: a signed 7x7 pattern adds
    # out-of-sample information.  It does not require one offset to be
    # individually significant, because several weak signed cells may only be
    # informative jointly.  The max-stat offset test is localization evidence.
    strong = bool(mean_effect > 0 and p_cv < 0.05)
    suggestive = bool(mean_effect > 0 and p_cv < 0.10)
    return {
        "observation": ("A signed 7x7 local score pattern contains information about intervention-derived "
                        "block utility beyond center score and key position."),
        "utility_definition": "baseline label NLL minus replacement label NLL; higher is better",
        "patch_definition": ("Exact Conv.py 7x7 cross-correlation patch with replicate padding; center is a "
                             "control and off-center weights may be positive or negative."),
        "primary_comparison": "signed full 7x7 ridge model versus center-score + key-position controls",
        "primary_statistic": ("Within each dependency group, average (control OOF MSE - signed-kernel "
                              "OOF MSE) / within-run utility variance, then average groups equally; "
                              "equivalent to dependency-balanced incremental OOF R-squared."),
        "run_count": int(len(runs)),
        "dependency_group_count": int(len(group_means)),
        "candidate_count": int(sum(len(run["keys"]) for run in runs)),
        "estimable_run_count": int(len(effects)),
        "folds": int(folds),
        "purge_radius_blocks": int(purge_radius),
        "ridge_alpha": float(ridge_alpha),
        "control_mode": control_mode,
        "mean_incremental_oof_r2": float(mean_effect),
        "median_incremental_oof_r2": float(np.median(effects)) if len(effects) else float("nan"),
        "runs_with_positive_oof_improvement": int(np.count_nonzero(effects > 0)),
        "dependency_group_bootstrap_95pct_ci": bootstrap_mean_ci(
            group_means, bootstrap_samples, seed + 1),
        "cv_permutation_test": {
            "method": permutation["method"],
            "requested_permutations": int(permutation["requested_permutations"]),
            "evaluated_null_shifts": int(permutation["evaluated_null_shifts"]),
            "one_sided_p_local_kernel_improves": p_cv,
            "null_quantiles": permutation["null_cv_effect_quantiles"],
        },
        "strongest_signed_offset": {
            "dq": int(best_row - radius),
            "dk": int(best_col - radius),
            "partial_r": max_corr,
            "familywise_corrected_p": float(permutation["offset_fwer_p"][best_row, best_col]),
        },
        "any_local_offset_max_stat_test": {
            "max_abs_partial_r": abs(max_corr),
            "familywise_p": p_offset,
            "null_quantiles": permutation["null_max_abs_correlation_quantiles"],
        },
        "decision_rule": ("Support requires positive mean incremental out-of-fold R-squared and a "
                          "one-sided synchronized circular-shift permutation p < 0.05. "
                          "Single-offset significance is reported but not required."),
        "evidence_grade": "strong" if strong else "suggestive" if suggestive else "not_supported",
        "supports_observation_in_this_dataset": strong,
        "scope_warning": ("Layers/heads from one prompt share data and are not independent replications. "
                          "Confirm on prespecified prompts, heads, and tasks before making a general claim."),
    }
