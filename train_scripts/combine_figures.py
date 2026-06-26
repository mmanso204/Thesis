#!/usr/bin/env python3
"""
Tile the four all-run overview figures into ONE page (2x2) and save as a single
PDF + PNG. Reads the PNG exports already produced by plot_results.py.

Usage:
    python train_scripts/combine_figures.py
    python train_scripts/combine_figures.py --fig_dir results/figures --stem overview_combined
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANELS = [
    "reward_all_runs",
    "delivery_all_runs",
    "completion_all_runs",
    "stage_progression",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig_dir", type=Path, default=Path("results/figures"))
    ap.add_argument("--stem", default="overview_combined")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, name in zip(axes.flat, PANELS):
        png = args.fig_dir / f"{name}.png"
        if not png.exists():
            ax.text(0.5, 0.5, f"missing:\n{name}.png", ha="center", va="center")
            ax.axis("off")
            continue
        ax.imshow(mpimg.imread(png))
        ax.axis("off")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.fig_dir / f"{args.stem}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
