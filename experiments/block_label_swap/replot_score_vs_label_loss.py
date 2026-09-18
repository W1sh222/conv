#!/usr/bin/env python3
"""Redraw only the initial-score versus label-loss scatter from saved results.

This script never loads the language model or reruns an intervention. It reads
the completed experiment metadata and trial records produced by
run_experiment.py, then writes a larger-marker plot with a framed legend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "directory",
        type=Path,
        help="Completed swap result directory containing experiment.json and trials.jsonl",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=80,
        help="Area of each candidate marker in points squared (default: 80)",
    )
    p.add_argument(
        "--baseline-size",
        type=float,
        default=150,
        help="Area of the orange baseline square (default: 150)",
    )
    p.add_argument(
        "--output-stem",
        default="score_vs_label_loss_framed",
        help="Output filename without extension; original plot is preserved by default",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p


def load_results(directory: Path):
    metadata_path = directory / "experiment.json"
    trials_path = directory / "trials.jsonl"
    if not metadata_path.is_file() or not trials_path.is_file():
        raise FileNotFoundError(
            f"Expected experiment.json and trials.jsonl in {directory}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("The saved sweep is incomplete; refusing to present it as a complete result")

    rows = [
        json.loads(line)
        for line in trials_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    plan = metadata["plan"]
    expected = plan["candidate_key_blocks"]
    actual = [row["key_block"] for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("trials.jsonl contains missing or duplicate candidate results")
    if not rows:
        raise ValueError("No candidate replacement results were found")
    return metadata, rows


def draw(directory: Path, point_size: float, baseline_size: float,
         output_stem: str, dpi: int):
    if point_size <= 0 or baseline_size <= 0 or dpi <= 0:
        raise ValueError("Marker sizes and dpi must be positive")
    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("output-stem must be a filename without directory components")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata, rows = load_results(directory)
    plan = metadata["plan"]
    baseline = float(metadata["baseline"]["label_loss"])
    arguments = metadata["arguments"]

    green = "#A9D6F5"
    green_edge = "#3F78A0"
    orange = "#F1AA66"
    orange_edge = "#774715"
    caption = (
        f"Layer {arguments['layer']} / Head {arguments['head']} / "
        f"Query block {plan['query_block']}"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(
        [row["initial_score"] for row in rows],
        [row["label_loss"] for row in rows],
        c=green,
        edgecolors=green_edge,
        linewidths=1.2,
        s=150,
        label="All candidates",
        zorder=3,
    )
    ax.scatter(
        [plan["removed_initial_score"]],
        [baseline],
        c=orange,
        edgecolors=orange_edge,
        linewidths=1.3,
        marker="s",
        s=150,
        label="Removed block",
        zorder=4,
    )
    ax.axhline(
        baseline,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.set_xlabel("Initial block score", fontsize=18)
    ax.set_ylabel("Ground-truth label loss", fontsize=18)
    ax.set_title(caption, fontsize=20)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=14,
    )
    legend = ax.legend(
        loc="lower right",
        fontsize=14,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#333333",
        borderpad=0.55,
    )
    legend.get_frame().set_linewidth(1.0)
    fig.tight_layout()

    png = directory / f"{output_stem}.png"
    pdf = directory / f"{output_stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    args = parser().parse_args()
    draw(
        args.directory,
        point_size=args.point_size,
        baseline_size=args.baseline_size,
        output_stem=args.output_stem,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
