#!/usr/bin/env python3
"""Re-render the original block-selection mask with only three colors."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Completed block-swap directory containing experiment.json and block_map.npz")
    p.add_argument("--output", required=True,
                   help="PNG output path; PDF and SVG files with the same stem are also written")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--font-scale", type=float, default=1.0,
                   help="Scale title, axis-label, tick, and legend fonts (default: 1.0)")
    p.add_argument("--legend-location", choices=("upper-right", "bottom"),
                   default="upper-right", help="Legend in the unused causal triangle or below the plot")
    return p


def draw(input_dir, output_png, dpi=300, font_scale=1.0,
         legend_location="upper-right"):
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
    if not math.isfinite(float(font_scale)) or float(font_scale) <= 0:
        raise ValueError("font_scale must be finite and positive")
    if legend_location not in ("upper-right", "bottom"):
        raise ValueError("legend_location must be upper-right or bottom")

    # 0: legal but unselected, 1: selected, 2: causally masked.
    image = np.full(selected.shape, 2, dtype=np.uint8)
    image[np.tri(*selected.shape, dtype=bool)] = 0
    image[selected] = 1

    args = meta.get("arguments", {})
    layer = args.get("layer", "?")
    head = args.get("head", "?")
    title = f"Layer {layer} / Head {head}"
    scale = float(font_scale)
    
    # 【修改点 1】：大幅调大所有基础字体数值
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 20 * scale,          # 原 14 (全局默认)
        "axes.labelsize": 28 * scale,     # 原 18 (X/Y轴标签: Key block / Query block)
        "xtick.labelsize": 24 * scale,    # 原 18 (X轴刻度数字)
        "ytick.labelsize": 24 * scale,    # 原 18 (Y轴刻度数字)
        "legend.fontsize": 20 * scale,    # 原 13 (图例文字: Selected 等)
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    blue, white, gray = "#92BEDF", "#FFFFFF", "#EEEEEE"
    # Keep a square map. Larger type is accommodated by a larger canvas rather
    # than reducing the map until the data become hard to read.
    canvas_scale = max(1.0, scale / 1.4)
    fig, ax = plt.subplots(figsize=(9 * canvas_scale, 9 * canvas_scale))
    ax.imshow(image, cmap=ListedColormap([white, blue, gray]), vmin=0, vmax=2,
              interpolation="nearest", origin="upper")
              
    # 【修改点 2】：增加 labelpad 防止大字体被边缘裁剪
    ax.set_xlabel("Key block", labelpad=15 * scale)
    ax.set_ylabel("Query block", labelpad=15 * scale)
    
    # 【修改点 3】：大幅调大主标题字体和顶部间距
    ax.set_title(title, fontsize=32 * scale, pad=20 * scale)
    
    # 增加刻度数字与坐标轴的间距
    ax.tick_params(direction="out", length=4, pad=8 * scale)
    
    handles = [
        Patch(facecolor=blue, edgecolor="#777777", linewidth=0.7, label="Selected"),
        Patch(facecolor=white, edgecolor="#777777", linewidth=0.7, label="Unselected"),
        Patch(facecolor=gray, edgecolor="#777777", linewidth=0.7, label="Causally masked"),
    ]
    legend_kwargs = dict(frameon=False, handlelength=1.35, handleheight=0.85,
                         handletextpad=0.6, labelspacing=0.75, borderaxespad=0)
    if legend_location == "upper-right":
        legend = ax.legend(handles=handles, loc="upper right",
                           bbox_to_anchor=(0.97, 0.97), ncol=1, **legend_kwargs)
        fig.tight_layout(pad=1.2)
        fig.canvas.draw()
        # The complete legend must remain within the gray, future-key triangle.
        # Its bottom-left is the corner closest to the causal diagonal.
        bounds = legend.get_window_extent(fig.canvas.get_renderer())
        left, bottom = ax.transAxes.inverted().transform((bounds.x0, bounds.y0))
        if left + bottom <= 1.02:
            plt.close(fig)
            raise ValueError("Legend would cover valid blocks; use --legend-location bottom")
    else:
        # A separate figure-level legend leaves a measured gap below the xlabel.
        legend = fig.legend(handles=handles, loc="lower center", ncol=3,
                            bbox_to_anchor=(0.5, 0.02), columnspacing=1.1,
                            **legend_kwargs)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if legend.get_window_extent(renderer).width > fig.bbox.width * 0.92:
            legend.remove()
            legend = fig.legend(handles=handles, loc="lower center", ncol=1,
                                bbox_to_anchor=(0.5, 0.02), **legend_kwargs)
            fig.canvas.draw()
        height = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
        fig.tight_layout(rect=(0, height + 0.05, 1, 1), pad=1.2)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output_png, output_pdf


def main():
    args = parser().parse_args()
    png, pdf = draw(args.input, args.output, args.dpi, args.font_scale,
                    args.legend_location)
    print(f"Saved {png}")
    print(f"Saved {pdf}")
    print(f"Saved {png.with_suffix('.svg')}")


if __name__ == "__main__":
    main()