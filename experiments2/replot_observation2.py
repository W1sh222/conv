#!/usr/bin/env python3
"""Re-render Observation 2 with the manuscript's blue/orange palette.

The script only reads the completed Observation 2 CSV/JSON outputs. It does
not rerun the model or recompute any candidate losses.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "acl_revision" / "Figs" / "observe2.png"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Completed observation2_line6 directory")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="PNG path (default: acl_revision/Figs/observe2.png); also writes PDF/SVG")
    p.add_argument("--point-size", type=float, default=30.0,
                   help="Candidate marker area in points squared (default: 30)")
    p.add_argument("--font-scale", type=float, default=1.0,
                   help="Scale all fonts and the figure dimensions (default: 1.0)")
    p.add_argument("--dpi", type=int, default=600)
    return p


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_float(row, key):
    return float(row[key])


def draw(input_dir, output_png=DEFAULT_OUTPUT, point_size=30.0, dpi=600,
         font_scale=1.0):
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
    scale = float(font_scale)
    if not math.isfinite(point_size) or point_size <= 0 or dpi <= 0:
        raise ValueError("point-size must be finite and positive, and dpi must be positive")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("font-scale must be finite and positive")
    if 8 * scale < 5:
        raise ValueError("font-scale must be at least 0.625 to keep every glyph at 5 pt or larger")
    point_size *= scale ** 2

    # A 183-mm-wide two-panel figure. Blue and orange match the manuscript's
    # candidate-block figure; the grouping encodes context, not attention method.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9 * scale,
        "axes.titlesize": 10 * scale,
        "axes.labelsize": 9 * scale,
        "xtick.labelsize": 8.5 * scale,
        "ytick.labelsize": 8.5 * scale,
        "legend.fontsize": 8 * scale,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": .75,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    blue, orange = "#A9D6F5", "#F1AA66"
    blue_edge, orange_edge, gray = "#3F78A0", "#774715", "#999999"
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4 * scale, 3.45 * scale))
    fig.subplots_adjust(left=.095, right=.985, bottom=.285, top=.84, wspace=.43)
    for letter, ax in zip(("a", "b"), axes):
        ax.tick_params(direction="out", length=3, width=.7, pad=3 * scale)
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-18 * scale, 12 * scale), textcoords="offset points",
                    fontsize=10 * scale, fontweight="bold", annotation_clip=False)

    # Left panel: the same center-score-matched pairs as the original figure.
    ax = axes[0]
    for pair in pairs:
        low_utility = as_float(pair, "low_utility")
        high_utility = as_float(pair, "high_utility")
        ax.plot([0, 1], [low_utility, high_utility], color=gray,
                alpha=.52, linewidth=.65, zorder=1)
        ax.scatter([0], [low_utility], s=point_size * .62, color=orange,
                   edgecolor=orange_edge, linewidth=.65, zorder=2)
        ax.scatter([1], [high_utility], s=point_size * .62, color=blue,
                   edgecolor=blue_edge, linewidth=.65, zorder=2)
    low_mean = sum(as_float(pair, "low_utility") for pair in pairs) / len(pairs)
    high_mean = sum(as_float(pair, "high_utility") for pair in pairs) / len(pairs)
    ax.plot([0, 1], [low_mean, high_mean], color="#333333", linewidth=1.5, zorder=3)
    ax.scatter([0, 1], [low_mean, high_mean], marker="D", s=point_size * 1.15,
               color="white", edgecolor="#333333", linewidth=1, zorder=4)
    ax.axhline(0, color="#AAAAAA", linestyle="--", linewidth=.7, zorder=0)
    ax.set_xticks([0, 1], ["Lower context", "Higher context"])
    ax.set_ylabel("Block utility")
    ax.set_xlim(-.12, 1.12)
    ax.set_title("Center-score-matched candidates", pad=12 * scale)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=orange,
               markeredgecolor=orange_edge, markersize=5 * scale, label="Lower context"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=blue,
               markeredgecolor=blue_edge, markersize=5 * scale, label="Higher context"),
        Line2D([0], [0], marker="D", linestyle="-", color="black",
               markerfacecolor="white", markersize=4.5 * scale, label="Pair mean"),
    ]
    # Dedicated space below each panel avoids concealing any measured points.
    # Two entries on the first row; the mean occupies the second row.
    ax.legend(handles=[handles[0], handles[2], handles[1]], ncol=2,
              frameon=False, loc="upper center",
              bbox_to_anchor=((ax.get_position().x0 + ax.get_position().x1)/2, .13),
              bbox_transform=fig.transFigure,
              columnspacing=1, handletextpad=.4, handlelength=1.4,
              borderaxespad=0, labelspacing=.6)

    # Right panel: residualized neighborhood score versus residualized utility.
    ax = axes[1]
    x = np.asarray([as_float(row, "context_residual") for row in candidates])
    y = np.asarray([as_float(row, "utility_residual") for row in candidates])
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x) == 0:
        raise ValueError("Residual data must be finite with nonconstant x values")
    ax.scatter(x, y, s=point_size, color=blue, edgecolor=blue_edge, linewidth=.65,
               alpha=.92, label="Candidates")
    coefficients = np.polyfit(x, y, 1)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    correlation = float(summary["partial_neighbor_pearson_r"])
    ax.plot(x_line, np.polyval(coefficients, x_line), color=blue_edge,
            linewidth=1.5, label=f"Linear fit (r = {correlation:.3f})")
    ax.axhline(0, color="#AAAAAA", linestyle="--", linewidth=.7, zorder=0)
    ax.axvline(0, color="#AAAAAA", linestyle="--", linewidth=.7, zorder=0)
    ax.set_xlabel("Neighborhood score residual")
    ax.set_ylabel("Block utility residual")
    ax.set_title("Center-adjusted local context", pad=12 * scale)
    ax.legend(frameon=False, loc="upper center",
              bbox_to_anchor=((ax.get_position().x0 + ax.get_position().x1)/2, .13),
              bbox_transform=fig.transFigure,
              ncol=1, handletextpad=.5, handlelength=1.6, borderaxespad=0,
              labelspacing=.6)

    # Deliberately no fig.suptitle: the requested top-level title is removed.
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.canvas.draw()
    # Always measure the final panel rectangles, including on servers where
    # the skill's optional audit module is not installed.
    rects = [list(ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).extents * 72)
             for ax in axes]
    for coordinate in (1, 3):
        if abs(rects[0][coordinate] - rects[1][coordinate]) > 1.5:
            raise RuntimeError("Final rendered panel edges differ by more than 1.5 pt")
    if abs((rects[0][2]-rects[0][0]) - (rects[1][2]-rects[1][0])) > 1.5:
        raise RuntimeError("Final rendered panel widths differ by more than 1.5 pt")
    layout = {"schema_version": 1, "backend": "matplotlib",
              "figure": {"width_pt": fig.get_figwidth()*72, "height_pt": fig.get_figheight()*72},
              "panels": [{"id": letter, "bbox_pt": bbox, "grid_id": "main",
                          "row_start": 0, "row_stop": 1, "col_start": i, "col_stop": i+1}
                         for i, (letter, bbox) in enumerate(zip(("a", "b"), rects))]}
    output_png.with_suffix(".alignment-layout.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        pass
    else:
        require_matplotlib_panel_alignment(fig, json_out=output_png.with_suffix(".alignment.json"),
                                           require_panel_labels=True, strict=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output_png, output_pdf


def main():
    args = parser().parse_args()
    png, pdf = draw(args.input, args.output, args.point_size, args.dpi, args.font_scale)
    print(f"Saved {png}")
    print(f"Saved {pdf}")
    print(f"Saved {png.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
