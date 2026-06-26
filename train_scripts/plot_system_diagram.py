#!/usr/bin/env python3
"""
Render the system-overview loop diagram for Chapter 4.1.

Shows the perception -> reasoning -> action loop:
  Environment -> observation -> Reasoning layer (ABox+TBox -> reasoner ->
  derived facts) -> {state augmentation, reward shaping} -> MAPPO -> action.

The dashed box marks the ontology layer, i.e. the only part removed in the
vanilla baseline.

Usage:
    python train_scripts/plot_system_diagram.py
    python train_scripts/plot_system_diagram.py --out_dir thesis/figures

Outputs: system_overview.pdf and .png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# palette
C_ENV   = "#dbe9f6"   # environment
C_REAS  = "#e7f4e4"   # reasoning layer
C_INJ   = "#fdf0d5"   # injection points
C_LEARN = "#f6dbe0"   # learner
EDGE    = "#333333"
ONT     = "#2ca02c"   # ontology-layer enclosure


def box(ax, cx, cy, w, h, text, color, fontsize=11, bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=EDGE, facecolor=color, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", zorder=3,
            fontsize=fontsize, fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, label=None, lblpos=None, rad=0.0, fs=9):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.4, color=EDGE, zorder=1,
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        lx, ly = lblpos if lblpos else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=fs,
                style="italic", color="#444444", zorder=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("thesis/figures"))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ontology-layer enclosure (drawn first, behind boxes)
    ax.add_patch(Rectangle((3.55, 1.75), 7.1, 4.7, fill=False,
                           linewidth=1.6, edgecolor=ONT, linestyle=(0, (6, 4)),
                           zorder=0))
    ax.text(7.1, 6.05, "Ontology layer  (removed in vanilla baseline)",
            ha="center", va="bottom", fontsize=9.5, color=ONT,
            fontweight="bold")

    # boxes
    box(ax, 1.6, 4.1, 2.4, 1.6, "Environment\n(MultiGrid)\n+ TBox (schema)", C_ENV,
        bold=True, fontsize=10)

    # reasoning layer with internal pipeline text
    box(ax, 6.0, 4.1, 3.4, 3.2, "", C_REAS)
    ax.text(6.0, 5.55, "Reasoning layer", ha="center", va="center",
            fontsize=11, fontweight="bold", zorder=3)
    for cy, txt in [(4.55, "ABox (from obs)  +  TBox (from env)"),
                    (3.65, "OWL reasoner"),
                    (2.75, "derived facts")]:
        ax.text(6.0, cy, txt, ha="center", va="center", fontsize=9.5, zorder=3)
    arrow(ax, (6.0, 4.32), (6.0, 3.92))
    arrow(ax, (6.0, 3.42), (6.0, 3.02))

    box(ax, 9.6, 5.25, 2.5, 1.0, "State\naugmentation", C_INJ, fontsize=10)
    box(ax, 9.6, 2.95, 2.5, 1.0, "Reward\nshaping", C_INJ, fontsize=10)

    box(ax, 13.2, 4.1, 2.4, 1.6, "MAPPO\n(actor + critic)", C_LEARN, bold=True)

    # arrows
    # two inputs from the environment: per-step observation builds the ABox,
    # while the (static) TBox schema is also supplied by the environment.
    arrow(ax, (2.8, 4.45), (4.3, 4.75), "observation\n(field of view)", (3.55, 5.15), fs=8)
    arrow(ax, (2.8, 3.75), (4.3, 4.35), "TBox\n(domain schema)", (3.55, 3.25), fs=8)
    arrow(ax, (7.0, 4.6), (8.35, 5.25))
    arrow(ax, (7.0, 3.6), (8.35, 2.95))
    arrow(ax, (10.85, 5.25), (12.1, 4.55), "augmented obs.", (11.9, 5.25), fs=8)
    arrow(ax, (10.85, 2.95), (12.1, 3.65), "shaped reward", (11.85, 2.65), fs=8)

    # native observation: Environment -> MAPPO directly. Always available; the
    # ontology layer augments this obs rather than replacing it (in the baseline
    # this direct path is all that remains).
    y_top = 7.0
    ax.plot([1.6, 1.6], [4.85, y_top], color=EDGE, lw=1.4, zorder=1)
    ax.plot([1.6, 13.2], [y_top, y_top], color=EDGE, lw=1.4, zorder=1)
    ax.add_patch(FancyArrowPatch(
        (13.2, y_top), (13.2, 4.9), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.4, color=EDGE, zorder=1))
    ax.text(7.4, y_top + 0.25, "native observation", ha="center", va="center",
            fontsize=9, style="italic", color="#444444")

    # return loop: MAPPO -> action -> Environment (explicit L-shaped routing)
    y_ret = 0.85
    ax.plot([13.2, 13.2], [3.3, y_ret], color=EDGE, lw=1.4, zorder=1)
    ax.plot([13.2, 1.6], [y_ret, y_ret], color=EDGE, lw=1.4, zorder=1)
    ax.add_patch(FancyArrowPatch(
        (1.6, y_ret), (1.6, 3.35), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.4, color=EDGE, zorder=1))
    ax.text(7.4, y_ret + 0.28, "action", ha="center", va="center", fontsize=9.5,
            style="italic", color="#444444")

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"system_overview.{ext}",
                    dpi=150, bbox_inches="tight")
    print(f"saved system_overview.pdf / .png to {args.out_dir}/")
    plt.close(fig)


if __name__ == "__main__":
    main()
