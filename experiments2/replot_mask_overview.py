#!/usr/bin/env python3
"""Re-render the original block-selection mask with only three colors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Completed block-swap directory containing experiment.json and block_map.npz")
    p.add_argument("--output", required=True,
                   help="PNG output path; a PDF with the same stem is also written")
    p.add_argument("--dpi", type=int, default=300)
    return p


def draw(input_dir, output_png, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    input_dir = Path(input_dir).resolve()
    output_png = Path(output_png).resolve()
    if output_png.suffix.lower() != ".png":
        raise ValueError("--output must be a .png path")
    meta_file = input_dir / "experiment.json"
    map_file = input_dir / "block_map.npz"
    if not meta_file.is_file() or not map_file.is_file():
        raise FileNotFoundError(
            f"Expected experiment.json and block_map.npz in {input_dir}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    arrays = np.load(map_file)
    selected = np.asarray(arrays["selected_mask"], dtype=bool)
    if selected.ndim != 2 or selected.shape[0] != selected.shape[1]:
        raise ValueError("selected_mask must be a square 2-D block mask")
    if np.any(selected & ~np.tri(*selected.shape, dtype=bool)):
        raise ValueError("selected_mask contains causally invalid future blocks")
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")

    # 0: legal but unselected, 1: selected, 2: causally masked.
    image = np.full(selected.shape, 2, dtype=np.uint8)
    image[np.tri(*selected.shape, dtype=bool)] = 0
    image[selected] = 1

    args = meta.get("arguments", {})
    layer = args.get("layer", "?")
    head = args.get("head", "?")
    title = f"Layer {layer} / Head {head}"
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    blue, white, gray = "#92BEDF", "#FFFFFF", "#EEEEEE"
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(image, cmap=ListedColormap([white, blue, gray]), vmin=0, vmax=2,
              interpolation="nearest", origin="upper")
    ax.set_xlabel("Key block")
    ax.set_ylabel("Query block")
    ax.set_title(title, fontsize=16)
    ax.legend(handles=[
        Patch(facecolor=blue, edgecolor="#444444", label="Selected"),
        Patch(facecolor=white, edgecolor="#777777", label="Unselected"),
        Patch(facecolor=gray, edgecolor="#EEEEEE", label="Causally masked"),
    ], loc="upper center", bbox_to_anchor=(.5, -.09), ncol=3,
               frameon=False)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return output_png, output_pdf


def main():
    args = parser().parse_args()
    png, pdf = draw(args.input, args.output, args.dpi)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
