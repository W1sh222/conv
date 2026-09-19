#!/usr/bin/env python3
"""Re-render Observation 2 with larger fonts and without the figure suptitle.

The script only reads the completed Observation 2 CSV/JSON outputs. It does
not rerun the model or recompute any candidate losses.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Completed observation2_line6 directory")
    p.add_argument("--output", required=True,
                   help="PNG output path; a PDF with the same stem is also written")
    p.add_argument("--point-size", type=float, default=75.0)
    p.add_argument("--dpi", type=int, default=300)
    return p


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_float(row, key):
    return float(row[key])


def draw(input_dir, output_png, point_size=75.0, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    input_dir = Path(input_dir).resolve()
    output_png = Path(output_png).resolve()
    if output_png.suffix.lower() != ".png":
        raise ValueError("--output must be a .png path")
    candidate_file = input_dir / "candidate_context.csv"
    pairs_file = input_dir / "matched_pairs.csv"
    summary_file = input_dir / "summary.json"
    for filename in (candidate_file, pairs_file, summary_file):
        if not filename.is_file():
            raise FileNotFoundError(f"Missing Observation 2 output: {filename}")
    candidates = read_csv(candidate_file)
    pairs = read_csv(pairs_file)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if not candidates or not pairs:
        raise ValueError("Observation 2 CSV files contain no data")
    point_size = float(point_size)
    dpi = int(dpi)
    if point_size <= 0 or dpi <= 0:
        raise ValueError("point-size and dpi must be positive")

    # The previous figure used Matplotlib's defaults (10 pt for labels/ticks,
    # 12 pt for axes titles, and 10 pt for legends). Each is increased by 2 pt.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.1,
    })
    green, orange, gray, dark = "#A8D7B0", "#F1AA66", "#9A9A9A", "#32633B"
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # Left panel: the same center-score-matched pairs as the original figure.
    ax = axes[0]
    tolerance = max(as_float(pair, "effective_tolerance") for pair in pairs)
    for pair in pairs:
        low_utility = as_float(pair, "low_utility")
        high_utility = as_float(pair, "high_utility")
        difference = high_utility - low_utility
        color = dark if difference > tolerance else "#A85C25" if difference < -tolerance else gray
        ax.plot([0, 1], [low_utility, high_utility], color=color,
                alpha=.48, linewidth=1.2, zorder=1)
        ax.scatter([0], [low_utility], s=point_size * .62, color=orange,
                   edgecolor="#774715", linewidth=.8, zorder=2)
        ax.scatter([1], [high_utility], s=point_size * .62, color=green,
                   edgecolor=dark, linewidth=.8, zorder=2)
    low_mean = sum(as_float(pair, "low_utility") for pair in pairs) / len(pairs)
    high_mean = sum(as_float(pair, "high_utility") for pair in pairs) / len(pairs)
    ax.plot([0, 1], [low_mean, high_mean], color="black", linewidth=2.4, zorder=3)
    ax.scatter([0, 1], [low_mean, high_mean], marker="D", s=point_size * 1.15,
               color="white", edgecolor="black", linewidth=1.5, zorder=4)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1], ["Lower context", "Higher context"])
    ax.set_ylabel("Block utility (baseline loss $-$ replacement loss)")
    ax.set_title("Center-score-matched candidates")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=orange,
               markeredgecolor="#774715", markersize=8, label="Lower context"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=green,
               markeredgecolor=dark, markersize=8, label="Higher context"),
        Line2D([0], [0], marker="D", linestyle="-", color="black",
               markerfacecolor="white", markersize=7, label="Pair mean"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white",
              edgecolor="#333333", framealpha=1.0, loc="best")

    # Right panel: residualized neighborhood score versus residualized utility.
    ax = axes[1]
    x = np.asarray([as_float(row, "context_residual") for row in candidates])
    y = np.asarray([as_float(row, "utility_residual") for row in candidates])
    ax.scatter(x, y, s=point_size, color=green, edgecolor=dark, linewidth=1.1,
               alpha=.92, label="Candidate blocks")
    coefficients = np.polyfit(x, y, 1)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    correlation = float(summary["partial_neighbor_pearson_r"])
    ax.plot(x_line, np.polyval(coefficients, x_line), color="#315A85",
            linewidth=2, label=f"Linear fit ($r={correlation:.3f}$)")
    ax.axhline(0, color="#999999", linestyle="--", linewidth=1)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("Neighborhood score residual\n(center-score effect removed)")
    ax.set_ylabel("Block utility residual\n(center-score effect removed)")
    ax.set_title("Additional information from local context")
    ax.legend(frameon=True, facecolor="white", edgecolor="#333333",
              framealpha=1.0, loc="best")

    # Deliberately no fig.suptitle: the requested top-level title is removed.
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return output_png, output_pdf


def main():
    args = parser().parse_args()
    png, pdf = draw(args.input, args.output, args.point_size, args.dpi)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
