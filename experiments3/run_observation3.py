#!/usr/bin/env python3
"""Observation 3: test whether a signed 7x7 score patch predicts block utility."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from local_kernel_core import (
    aggregate_incremental_r2_effect,
    evaluate_cv,
    feature_indices,
    load_run,
    permutation_tests,
    pooled_offset_correlations,
    prepare_cv_plans,
    prepare_offset_state,
    summarize_analysis,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True,
                   help="Complete swap directories; quoted glob patterns are accepted")
    p.add_argument("--output", required=True, help="New or empty output directory")
    p.add_argument("--kernel-size", type=int, default=7)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--purge-radius", type=int, default=6,
                   help="Remove training centers this many key blocks from each test center")
    p.add_argument("--ridge-alpha", type=float, default=0.1,
                   help="Fixed prespecified L2 penalty after feature standardization")
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point-size", type=float, default=38.0)
    p.add_argument("--no-plots", action="store_true")
    return p


def natural_key(path):
    return [int(piece) if piece.isdigit() else piece.lower()
            for piece in re.split(r"(\d+)", str(path))]


def expand_inputs(arguments):
    paths = []
    for item in arguments:
        matches = glob.glob(item)
        paths.extend(matches if matches else [item])
    unique = []
    seen = set()
    for item in sorted(paths, key=natural_key):
        resolved = str(Path(item).resolve())
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def collect_kernel_rows(runs, results):
    rows = []
    size = int(runs[0]["kernel_size"])
    radius = size // 2
    for run, result in zip(runs, results):
        for fitted in result["kernels"]:
            for row in range(size):
                for col in range(size):
                    rows.append({
                        "run_id": run["run_id"], "layer": run["layer"],
                        "head": run["head"], "fold": fitted["fold"],
                        "dq": row - radius, "dk": col - radius,
                        "weight": float(fitted["kernel"][row, col]),
                        "normalized_weight": float(fitted["normalized_kernel"][row, col]),
                    })
    return rows


def kernel_summary(results):
    fitted = [item for result in results for item in result["kernels"]]
    kernels = np.asarray([item["normalized_kernel"] for item in fitted])
    mean = kernels.mean(axis=0)
    sign = np.sign(kernels)
    sign_stability = np.maximum(np.mean(sign > 0, axis=0), np.mean(sign < 0, axis=0))
    off_center = np.ones(mean.shape, dtype=bool)
    radius = mean.shape[0] // 2
    off_center[radius, radius] = False
    return {
        "fitted_kernel_count": len(fitted),
        "mean_normalized_signed_kernel": mean,
        "offset_sign_stability": sign_stability,
        "mean_positive_weight_count": float(np.mean([x["positive_count"] for x in fitted])),
        "mean_negative_weight_count": float(np.mean([x["negative_count"] for x in fitted])),
        "positive_mean_off_center_cells": int(np.count_nonzero(mean[off_center] > 0)),
        "negative_mean_off_center_cells": int(np.count_nonzero(mean[off_center] < 0)),
    }


def observed_analysis(runs, folds, purge, alpha, control, subset, return_kernels=False):
    plans = [prepare_cv_plans(run, folds, purge, alpha, control, subset) for run in runs]
    results = [evaluate_cv(run, plan, return_kernels=return_kernels)
               for run, plan in zip(runs, plans)]
    mean, effects = aggregate_incremental_r2_effect(results, runs)
    return plans, results, mean, effects


def make_plots(output, runs, results, correlations, offset_rows, kernels, summary, point_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": 1.0})
    blue, green, orange = "#315A85", "#4A8557", "#D4843E"
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))

    ax = axes[0, 0]
    x = np.arange(len(runs))
    effects = np.asarray([result["metrics"]["incremental_oof_r2"] for result in results])
    labels = [str(run["layer"]) if run["layer"] >= 0 else run["run_id"] for run in runs]
    colors = np.where(effects >= 0, green, orange)
    ax.bar(x, 100 * effects, color=colors, edgecolor="#333333", linewidth=.5)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    if len(x) <= 24:
        ax.set_xticks(x, labels, rotation=60 if len(x) > 12 else 0)
    else:
        tick = np.linspace(0, len(x) - 1, min(12, len(x)), dtype=int)
        ax.set_xticks(tick, [labels[i] for i in tick], rotation=45)
    ax.set_xlabel("Layer" if all(run["layer"] >= 0 for run in runs) else "Run")
    ax.set_ylabel("Incremental OOF $R^2$ (percentage points)")
    p_value = summary["cv_permutation_test"]["one_sided_p_local_kernel_improves"]
    aggregate = summary["mean_incremental_oof_r2"]
    ax.set_title(f"Signed 7x7 gain by run (group-balanced mean={100 * aggregate:.2f}%, p={p_value:.4g})")

    ax = axes[0, 1]
    actual, predicted = [], []
    for run, result in zip(runs, results):
        valid = result["valid"]
        y = run["utility"][valid]
        pred = result["full_prediction"][valid]
        scale = float(np.std(y)) or 1.0
        actual.extend((y - y.mean()) / scale)
        predicted.extend((pred - y.mean()) / scale)
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    ax.scatter(predicted, actual, s=point_size, color="#A8D7B0", edgecolor=green,
               linewidth=.7, alpha=.78)
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))
    ax.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Out-of-fold predicted utility (within-run SD)")
    ax.set_ylabel("Observed utility (within-run SD)")
    ax.set_title("Held-out prediction; each point is one candidate block")

    limit = float(max(np.max(np.abs(kernels)), 1e-12))
    ax = axes[1, 0]
    image = ax.imshow(kernels, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    radius = kernels.shape[0] // 2
    ticks = np.arange(kernels.shape[0])
    labels2 = [str(i - radius) for i in ticks]
    ax.set_xticks(ticks, labels2)
    ax.set_yticks(ticks, labels2)
    ax.set_xlabel("Key-block offset $\\Delta k$")
    ax.set_ylabel("Query-block offset $\\Delta q$")
    ax.set_title("Mean normalized fitted kernel (signed)")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)

    ax = axes[1, 1]
    corr_limit = float(max(np.nanmax(np.abs(correlations)), 1e-12))
    image = ax.imshow(correlations, cmap="RdBu_r", vmin=-corr_limit, vmax=corr_limit,
                      interpolation="nearest")
    ax.set_xticks(ticks, labels2)
    ax.set_yticks(ticks, labels2)
    ax.set_xlabel("Key-block offset $\\Delta k$")
    ax.set_ylabel("Query-block offset $\\Delta q$")
    ax.set_title("Signed partial correlation (center/position removed)")
    for record in offset_rows:
        if record["dq"] == 0 and record["dk"] == 0:
            continue
        if record["fwer_p"] < .05:
            ax.text(record["dk"] + radius, record["dq"] + radius, "*",
                    ha="center", va="center", fontsize=13, color="black")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)

    fig.suptitle("Observation 3: Signed 7x7 local pattern predicts block utility", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .97))
    for extension in ("png", "pdf"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output / f"observation3.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main():
    args = parser().parse_args()
    if args.kernel_size < 3 or args.kernel_size % 2 == 0:
        raise ValueError("kernel-size must be an odd integer >= 3")
    if args.folds < 2 or args.purge_radius < 0 or args.ridge_alpha < 0:
        raise ValueError("folds >= 2, purge-radius >= 0, and ridge-alpha >= 0 are required")
    if args.permutations < 100 or args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 permutations and bootstrap samples")
    if args.point_size <= 0:
        raise ValueError("point-size must be positive")

    inputs = expand_inputs(args.input)
    runs = [load_run(path, args.kernel_size, f"run_{index:03d}")
            for index, path in enumerate(inputs)]
    if not runs:
        raise ValueError("No input runs found")
    runs.sort(key=lambda run: (run["layer"], run["head"], run["source"]))
    for index, run in enumerate(runs):
        run["run_id"] = f"run_{index:03d}"

    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}")
    if not args.no_plots:
        import matplotlib  # noqa: F401
    output.mkdir(parents=True, exist_ok=True)

    plans, results, observed_effect, effects = observed_analysis(
        runs, args.folds, args.purge_radius, args.ridge_alpha,
        "center_position", "full", return_kernels=True)
    offset_states = [prepare_offset_state(run, "center_position") for run in runs]
    correlations = pooled_offset_correlations(runs, offset_states)
    permutation = permutation_tests(
        runs, plans, offset_states, observed_effect, correlations,
        args.permutations, args.seed)
    summary = summarize_analysis(
        runs, results, correlations, permutation, args.ridge_alpha, args.folds,
        args.purge_radius, "center_position", args.bootstrap_samples, args.seed)

    _, center_results, center_effect, center_effects = observed_analysis(
        runs, args.folds, args.purge_radius, args.ridge_alpha,
        "center", "full", return_kernels=False)
    summary["secondary_center_score_only_control"] = {
        "mean_incremental_oof_r2": center_effect,
        "runs_with_positive_improvement": int(np.count_nonzero(center_effects > 0)),
        "description": "Same signed 7x7 model, with only center score in the control model.",
    }

    ablation_rows = []
    for subset in ("full", "vertical_diag", "same_row", "same_column", "diagonals"):
        if subset == "full":
            subset_results, mean, subset_effects = results, observed_effect, effects
        else:
            _, subset_results, mean, subset_effects = observed_analysis(
                runs, args.folds, args.purge_radius, args.ridge_alpha,
                "center_position", subset, return_kernels=False)
        ablation_rows.append({
            "feature_subset": subset,
            "feature_count": len(feature_indices(args.kernel_size, subset)),
            "mean_incremental_oof_r2": mean,
            "median_incremental_oof_r2": float(np.median(subset_effects)),
            "runs_with_positive_improvement": int(np.count_nonzero(subset_effects > 0)),
            "run_count": len(subset_effects),
        })
    summary["spatial_ablations"] = ablation_rows

    alpha_values = sorted(set([.001, .01, .1, 1.0, 10.0, float(args.ridge_alpha)]))
    alpha_rows = []
    for alpha in alpha_values:
        if math.isclose(alpha, args.ridge_alpha, rel_tol=0, abs_tol=1e-15):
            mean, alpha_effects = observed_effect, effects
        else:
            _, _, mean, alpha_effects = observed_analysis(
                runs, args.folds, args.purge_radius, alpha,
                "center_position", "full", return_kernels=False)
        alpha_rows.append({
            "ridge_alpha": alpha, "mean_incremental_oof_r2": mean,
            "median_incremental_oof_r2": float(np.median(alpha_effects)),
            "runs_with_positive_improvement": int(np.count_nonzero(alpha_effects > 0)),
            "run_count": len(alpha_effects),
            "is_primary": bool(math.isclose(alpha, args.ridge_alpha, rel_tol=0, abs_tol=1e-15)),
        })
    summary["ridge_sensitivity"] = alpha_rows

    fitted_summary = kernel_summary(results)
    summary["signed_kernel_summary"] = fitted_summary
    summary["permutation_dependency_handling"] = {
        "group_count": permutation["dependency_group_count"],
        "description": permutation["dependency_handling"],
    }
    summary["arguments"] = vars(args)
    summary["expanded_inputs"] = inputs
    summary["input_runs"] = [{
        "run_id": run["run_id"], "directory": run["source"], "layer": run["layer"],
        "head": run["head"], "query_block": run["query_block"],
        "candidate_count": len(run["keys"]),
    } for run in runs]
    summary["status"] = "complete"
    summary["source_sha256"] = {
        "run_observation3.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "local_kernel_core.py": hashlib.sha256(
            (Path(__file__).parent / "local_kernel_core.py").read_bytes()).hexdigest(),
    }

    per_run_rows, prediction_rows, candidate_rows = [], [], []
    size, radius = args.kernel_size, args.kernel_size // 2
    patch_fields = [f"score_dq{dq:+d}_dk{dk:+d}"
                    for dq in range(-radius, radius + 1)
                    for dk in range(-radius, radius + 1)]
    for run, result in zip(runs, results):
        row = {"run_id": run["run_id"], "layer": run["layer"], "head": run["head"],
               "query_block": run["query_block"], "source": run["source"]}
        row.update(result["metrics"])
        per_run_rows.append(row)
        for index, key in enumerate(run["keys"]):
            prediction_rows.append({
                "run_id": run["run_id"], "layer": run["layer"], "head": run["head"],
                "query_block": run["query_block"], "key_block": int(key),
                "utility": float(run["utility"][index]),
                "control_prediction": float(result["base_prediction"][index]),
                "signed_7x7_prediction": float(result["full_prediction"][index]),
                "control_squared_error": float((run["utility"][index] - result["base_prediction"][index]) ** 2),
                "signed_7x7_squared_error": float((run["utility"][index] - result["full_prediction"][index]) ** 2),
            })
            candidate = {
                "run_id": run["run_id"], "layer": run["layer"], "head": run["head"],
                "query_block": run["query_block"], "key_block": int(key),
                "center_score": float(run["center"][index]),
                "baseline_loss": run["baseline_loss"],
                "replacement_loss": float(run["replacement_loss"][index]),
                "utility": float(run["utility"][index]), "source": run["source"],
            }
            candidate.update(dict(zip(patch_fields, run["patches"][index].ravel())))
            candidate_rows.append(candidate)

    kernel_rows = collect_kernel_rows(runs, results)
    mean_kernel = np.asarray(fitted_summary["mean_normalized_signed_kernel"])
    stability = np.asarray(fitted_summary["offset_sign_stability"])
    offset_p = np.asarray(permutation["offset_fwer_p"])
    offset_rows = []
    for row in range(size):
        for col in range(size):
            is_center = bool(row == radius and col == radius)
            offset_rows.append({
                "dq": row - radius, "dk": col - radius,
                "partial_r": None if is_center else float(correlations[row, col]),
                "fwer_p": None if is_center else float(offset_p[row, col]),
                "mean_normalized_fitted_weight": float(mean_kernel[row, col]),
                "fold_sign_stability": float(stability[row, col]),
                "is_center": is_center,
            })

    write_csv(output / "per_run_metrics.csv", per_run_rows, list(per_run_rows[0]))
    write_csv(output / "oof_predictions.csv", prediction_rows, list(prediction_rows[0]))
    write_csv(output / "candidate_patches.csv", candidate_rows,
              [key for key in candidate_rows[0] if key not in patch_fields] + patch_fields)
    write_csv(output / "kernel_weights.csv", kernel_rows, list(kernel_rows[0]))
    write_csv(output / "offset_correlations.csv", offset_rows, list(offset_rows[0]))
    write_csv(output / "ablation_metrics.csv", ablation_rows, list(ablation_rows[0]))
    write_csv(output / "alpha_sensitivity.csv", alpha_rows, list(alpha_rows[0]))
    (output / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plots:
        make_plots(output, runs, results, correlations, offset_rows, mean_kernel,
                   json_safe(summary), args.point_size)

    compact = {key: summary[key] for key in (
        "run_count", "candidate_count", "mean_incremental_oof_r2",
        "runs_with_positive_oof_improvement", "cv_permutation_test",
        "strongest_signed_offset", "evidence_grade", "supports_observation_in_this_dataset")}
    print(json.dumps(json_safe(compact), ensure_ascii=False, indent=2))
    print(f"Saved Observation 3 analysis to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
