#!/usr/bin/env python3
"""Test whether local score context explains block utility beyond center score."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from context_core import load_run, match_by_center, summarize


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True,
                   help="One or more complete experiments/block_label_swap result directories")
    p.add_argument("--output", required=True, help="New/empty output directory")
    p.add_argument("--line-radius", type=int, default=3,
                   help="Number of blocks on each side of the vertical and "
                        "backslash-diagonal neighborhood lines")
    p.add_argument("--score-caliper", type=float, default=0.15,
                   help="Maximum matched center-score gap, in within-run standard deviations")
    p.add_argument("--min-pairs", type=int, default=8)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--bootstrap-samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point-size", type=float, default=75.0)
    p.add_argument("--no-plots", action="store_true")
    return p


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output, residual_records, pairs, summary, point_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": 1.1})
    green, orange, gray, dark = "#A8D7B0", "#F1AA66", "#9A9A9A", "#32633B"
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    ax = axes[0]
    tolerance = max(p["effective_tolerance"] for p in pairs)
    for pair in pairs:
        difference = pair["utility_difference"]
        color = dark if difference > tolerance else "#A85C25" if difference < -tolerance else gray
        ax.plot([0, 1], [pair["low_utility"], pair["high_utility"]],
                color=color, alpha=.48, linewidth=1.2, zorder=1)
        ax.scatter([0], [pair["low_utility"]], s=point_size * .62, color=orange,
                   edgecolor="#774715", linewidth=.8, zorder=2)
        ax.scatter([1], [pair["high_utility"]], s=point_size * .62, color=green,
                   edgecolor=dark, linewidth=.8, zorder=2)
    low_mean = sum(p["low_utility"] for p in pairs) / len(pairs)
    high_mean = sum(p["high_utility"] for p in pairs) / len(pairs)
    ax.plot([0, 1], [low_mean, high_mean], color="black", linewidth=2.4, zorder=3)
    ax.scatter([0, 1], [low_mean, high_mean], marker="D", s=point_size * 1.15,
               color="white", edgecolor="black", linewidth=1.5, zorder=4)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1], ["Lower context", "Higher context"])
    ax.set_ylabel("Block utility (baseline loss $-$ replacement loss)")
    ax.set_title("Center-score-matched candidates")
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=orange,
                      markeredgecolor="#774715", markersize=8, label="Lower context"),
               Line2D([0], [0], marker="o", linestyle="", markerfacecolor=green,
                      markeredgecolor=dark, markersize=8, label="Higher context"),
               Line2D([0], [0], marker="D", linestyle="-", color="black",
                      markerfacecolor="white", markersize=7, label="Pair mean")]
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="#333333",
              framealpha=1.0, loc="best")

    ax = axes[1]
    x = [r["context_residual"] for r in residual_records]
    y = [r["utility_residual"] for r in residual_records]
    ax.scatter(x, y, s=point_size, color=green, edgecolor=dark, linewidth=1.1,
               alpha=.92, label="Candidate blocks")
    import numpy as np
    coefficients = np.polyfit(x, y, 1)
    x_line = np.linspace(min(x), max(x), 200)
    ax.plot(x_line, np.polyval(coefficients, x_line), color="#315A85", linewidth=2,
            label=f"Linear fit ($r={summary['partial_neighbor_pearson_r']:.3f}$)")
    ax.axhline(0, color="#999999", linestyle="--", linewidth=1)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("Neighborhood score residual\n(center-score effect removed)")
    ax.set_ylabel("Block utility residual\n(center-score effect removed)")
    ax.set_title("Additional information from local context")
    ax.legend(frameon=True, facecolor="white", edgecolor="#333333",
              framealpha=1.0, loc="best")

    fig.suptitle("Observation 2: Local score context and block utility", fontsize=15)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output / f"observation2.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main():
    args = parser().parse_args()
    if args.line_radius < 1:
        raise ValueError("line-radius must be a positive integer")
    if args.score_caliper <= 0 or args.min_pairs < 2:
        raise ValueError("score-caliper must be positive and min-pairs >= 2")
    if args.permutations < 100 or args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 permutations/bootstrap samples")
    if args.point_size <= 0:
        raise ValueError("point-size must be positive")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}")
    if not args.no_plots:
        import matplotlib  # noqa: F401
    output.mkdir(parents=True, exist_ok=True)

    runs, records, pairs = [], [], []
    for index, directory in enumerate(args.input):
        run = load_run(directory, args.line_radius, run_id=f"run_{index:03d}")
        runs.append(run)
        records.extend(run["records"])
        run_pairs = match_by_center(run["records"], args.score_caliper)
        for pair_index, pair in enumerate(run_pairs):
            pair["pair_id"] = f"run_{index:03d}_pair_{pair_index:03d}"
        pairs.extend(run_pairs)
    if len(pairs) < args.min_pairs:
        raise ValueError(f"Only {len(pairs)} center-matched pairs; need {args.min_pairs}. "
                         "Increase --score-caliper or add complete input runs.")

    result, residual = summarize(records, pairs, args.permutations,
                                 args.bootstrap_samples, args.seed)
    result.update({
        "arguments": vars(args),
        "status": "complete",
        "input_runs": [{"run_id": run["run_id"], "directory": run["directory"],
                        "layer": run["meta"]["arguments"]["layer"],
                        "head": run["meta"]["arguments"]["head"],
                        "query_block": run["meta"]["plan"]["query_block"],
                        "candidate_count": len(run["records"])} for run in runs],
        "matching": ("Within each run, split candidates at median neighborhood mean; greedily pair "
                     "opposite groups by closest center score within the prespecified caliper. "
                     "Grouping and matching do not use label loss."),
        "source_sha256": {str(Path(__file__).name): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "context_core.py": hashlib.sha256((Path(__file__).parent / "context_core.py").read_bytes()).hexdigest()},
    })
    candidate_fields = ["run_id", "query_block", "key_block", "center_score", "neighbor_mean",
                        "neighbor_std", "neighbor_count", "replacement_loss", "baseline_loss",
                        "utility", "delta_loss", "effective_tolerance", "improves_baseline",
                        "context_residual", "utility_residual", "source"]
    pair_fields = ["pair_id", "run_id", "low_key_block", "high_key_block",
                   "low_center_score", "high_center_score", "center_z_gap",
                   "low_neighbor_mean", "high_neighbor_mean", "neighbor_gap",
                   "low_loss", "high_loss", "low_utility", "high_utility",
                   "utility_difference", "effective_tolerance", "outcome"]
    write_csv(output / "candidate_context.csv", residual, candidate_fields)
    write_csv(output / "matched_pairs.csv", pairs, pair_fields)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plots:
        make_plots(output, residual, pairs, result, args.point_size)
    print(json.dumps({key: result[key] for key in ["candidate_count", "matched_pair_count",
          "mean_utility_difference_high_minus_low_context", "partial_neighbor_pearson_r",
          "supports_observation_in_this_dataset"]}, ensure_ascii=False, indent=2))
    print(f"Saved Observation 2 analysis to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
