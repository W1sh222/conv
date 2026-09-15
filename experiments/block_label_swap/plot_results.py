#!/usr/bin/env python3
"""Render all measured replacements; no filtering for favorable outcomes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np


def plot_results(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    directory = Path(directory)
    meta = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete":
        raise ValueError("Refusing to plot an incomplete sweep as a complete result")
    rows = [json.loads(s) for s in (directory / "trials.jsonl").read_text().splitlines() if s.strip()]
    data = np.load(directory / "block_map.npz")
    scores, mask = data["initial_scores"], data["selected_mask"]
    plan, summary = meta["plan"], meta["summary"]
    qb, removed = plan["query_block"], plan["removed_key_block"]
    by_key = {r["key_block"]: r for r in rows}
    if set(by_key) != set(plan["candidate_key_blocks"]) or len(rows) != len(by_key):
        raise ValueError("Missing or duplicate replacement results")
    baseline = meta["baseline"]["label_loss"]
    blue, orange, green, gray = "#92BEDF", "#F1AA66", "#A8D7B0", "#EEEEEE"
    legend = [Patch(facecolor=blue, edgecolor="#444444", label="Kept"),
              Patch(facecolor=orange, edgecolor="#444444", label="Removed (baseline loss)"),
              Patch(facecolor=green, edgecolor="#444444", label="Candidate (replacement loss)")]
    caption = f"Layer {meta['arguments']['layer']} / Head {meta['arguments']['head']} / Query block {qb}"
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

    # Whole head overview. Only the chosen query row contains intervention results.
    image = np.full(mask.shape, 3, dtype=np.uint8)
    image[np.tri(*mask.shape, dtype=bool)] = 0
    image[mask] = 1
    image[qb, plan["candidate_key_blocks"]] = 2
    image[qb, removed] = 4
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(image, cmap=ListedColormap(["#FFFFFF", blue, green, gray, orange]),
              vmin=0, vmax=4, interpolation="nearest", origin="upper")
    ax.add_patch(Rectangle((-.5, qb-.5), mask.shape[1], 1, fill=False, edgecolor="#222222", lw=1.4))
    if mask.shape[0] <= 24:
        ax.set_xticks(np.arange(mask.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(mask.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="#777777", lw=.35)
    ax.set(xlabel="Key block", ylabel="Query block", title=caption)
    ax.legend(handles=legend+[Patch(facecolor=gray, label="Causally masked"),
                             Patch(facecolor="white", edgecolor="#777777", label="Untested unselected rows")],
              loc="upper center", bbox_to_anchor=(.5, -.09), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(directory / "mask_overview.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / "mask_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    # Every legal block is present. Paginate long contexts so all loss annotations
    # remain legible; cells are a key-index listing of ONE row, not new query rows.
    per_page, columns = 72, 12
    keys = list(range(qb+1))
    pages = math.ceil(len(keys)/per_page)
    for page in range(pages):
        page_keys = keys[page*per_page:(page+1)*per_page]
        nr = math.ceil(len(page_keys)/columns)
        nc = min(columns, len(page_keys))
        fig, ax = plt.subplots(figsize=(max(7.5, nc*1.35), nr*1.2+2.1))
        ax.set_xlim(0, nc)
        ax.set_ylim(nr, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        for pos, key in enumerate(page_keys):
            y, x = divmod(pos, columns)
            if key == removed:
                color, value = orange, baseline
            elif mask[qb, key]:
                color, value = blue, None
            else:
                color, value = green, by_key[key]["label_loss"]
            ax.add_patch(Rectangle((x+.025, y+.025), .95, .95, facecolor=color,
                                  edgecolor="#555555", linewidth=.7))
            ax.text(x+.5, y+.19, f"K{key}", ha="center", va="center", fontsize=10)
            ax.text(x+.5, y+.48, "Kept" if value is None else f"{value:.7f}",
                    ha="center", va="center", fontsize=10, weight="bold")
            ax.text(x+.5, y+.77, f"S={scores[qb,key]:.4g}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{caption}   |   page {page+1}/{pages}\n"
                     f"Label loss (lower is better); baseline = {baseline:.7f}", fontsize=12, pad=14)
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.08), ncol=3, frameon=False, fontsize=9)
        fig.tight_layout()
        name = "swap_losses" if pages == 1 else f"swap_losses_{page+1:03d}"
        fig.savefig(directory / (name+".png"), dpi=220, bbox_inches="tight")
        fig.savefig(directory / (name+".pdf"), bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter([r["initial_score"] for r in rows], [r["label_loss"] for r in rows],
               c=green, edgecolors="#32633B", s=36, label="All candidates")
    ax.scatter([plan["removed_initial_score"]], [baseline], c=orange, edgecolors="#774715",
               marker="s", s=85, label="Removed block / baseline")
    ax.axhline(baseline, color="#777777", linestyle="--", linewidth=1)
    ax.set(xlabel="Initial block score", ylabel="Ground-truth label loss", title=caption)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "score_vs_label_loss.png", dpi=220)
    fig.savefig(directory / "score_vs_label_loss.pdf")
    plt.close(fig)
    print(f"Saved overview, {pages} annotated block page(s), and score/loss scatter to {directory}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    plot_results(p.parse_args().directory)
