#!/usr/bin/env python3
"""
Plot training results for all thesis runs.

Reads training_log.csv from each checkpoint directory and produces
comparison figures for SQ1 (ont vs vanilla), SQ3 (shared vs independent ABox),
and SQ5 (trash generalisation), plus overview charts.

Usage (from repo root or train_scripts/):
    python train_scripts/plot_results.py
    python train_scripts/plot_results.py --ckpt_root /path/to/dir --out_dir results/figs
    python train_scripts/plot_results.py --smooth 100   # narrower smoothing window

Outputs saved to: results/figures/  (both .pdf and .png)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

# Run definitions: must match run_experiments.sh
RUNS_META = [
    dict(name="run1_ont_food_prox5",  label="Ont + Shared (run1)",  color="#1f77b4", ls="-",   goal="collect_food"),
    dict(name="run2_vanilla_food",     label="Vanilla (run2)",        color="#ff7f0e", ls="--",  goal="collect_food"),
    dict(name="run3_ont_food_prox0",   label="Ont + Indep (run3)",    color="#2ca02c", ls="-.",  goal="collect_food"),
    dict(name="run4_ont_trash_prox5",  label="Ont + Trash (run4)",    color="#9467bd", ls=":",   goal="collect_trash"),
]

# Number of active items per stage per goal (stages 1-4)
STAGE_ITEMS = {
    "collect_food":  {1: 1, 2: 2, 3: 4, 4: 8},
    "collect_trash": {1: 1, 2: 2, 3: 4, 4: 16},
}

_HERE = Path(__file__).parent
_DEFAULT_CKPT_ROOT = _HERE
_DEFAULT_OUT_DIR   = Path("results/figures")
_DEFAULT_SMOOTH    = 200


# Data loading
def load_run(ckpt_root: Path, run_name: str, max_steps: int | None = None) -> pd.DataFrame | None:
    csv_path = ckpt_root / f"checkpoints_{run_name}" / "training_log.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)
    df = df.sort_values("steps").reset_index(drop=True)
    # Clip to a matched budget so over-trained runs (run1/run2 ran past 20M) are
    # compared on the same footing as the fresh 20M runs.
    if max_steps is not None:
        df = df[df["steps"] <= max_steps].reset_index(drop=True)
    return df


def stage_first_steps(df: pd.DataFrame) -> dict[int, int]:
    """Return {stage: first env-step when that stage was active}."""
    out = {}
    for stage in sorted(df["stage"].unique()):
        out[int(stage)] = int(df.loc[df["stage"] == stage, "steps"].iloc[0])
    return out


# Smoothing
def smooth(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win, min_periods=1, center=True).mean()


def delivery_frac(df: pd.DataFrame, goal: str, win: int) -> pd.Series:
    stage_map = STAGE_ITEMS[goal]
    n_active = df["stage"].map(stage_map).fillna(1)
    return smooth(df["balls"] / n_active, win)


# Figure helpers
def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  {stem}.pdf / .png")
    plt.close(fig)


def _apply_defaults(ax: plt.Axes, title: str, xlabel: str, ylabel: str,
                    pct_y: bool = False) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if pct_y:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))


# Individual figures
def fig_reward_all(run_data: list[dict], out_dir: Path, win: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for rd in run_data:
        m = rd["meta"]
        s = smooth(rd["df"]["avg50"], win)
        ax.plot(rd["df"]["steps"] / 1e6, s, label=m["label"],
                color=m["color"], linestyle=m["ls"], lw=1.8)
    _apply_defaults(ax, "Training reward: all runs",
                    "Environment steps (M)", f"Avg-50 reward (smoothed, win={win})")
    fig.tight_layout()
    _save(fig, out_dir, "reward_all_runs")


def fig_delivery_all(run_data: list[dict], out_dir: Path, win: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    food_runs  = [rd for rd in run_data if rd["meta"]["goal"] == "collect_food"]
    trash_runs = [rd for rd in run_data if rd["meta"]["goal"] == "collect_trash"]

    for rd in food_runs:
        m = rd["meta"]
        frac = delivery_frac(rd["df"], "collect_food", win)
        axes[0].plot(rd["df"]["steps"] / 1e6, frac, label=m["label"],
                     color=m["color"], linestyle=m["ls"], lw=1.8)
    _apply_defaults(axes[0], "Delivery fraction: collect_food",
                    "Steps (M)", "Delivered / active items", pct_y=True)

    for rd in trash_runs:
        m = rd["meta"]
        frac = delivery_frac(rd["df"], "collect_trash", win)
        axes[1].plot(rd["df"]["steps"] / 1e6, frac, label=m["label"],
                     color=m["color"], linestyle=m["ls"], lw=1.8)
    _apply_defaults(axes[1], "Delivery fraction: collect_trash (SQ5)",
                    "Steps (M)", "Delivered / active items", pct_y=True)

    fig.tight_layout()
    _save(fig, out_dir, "delivery_all_runs")


def fig_completion_all(run_data: list[dict], out_dir: Path, win: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for rd in run_data:
        m = rd["meta"]
        rate = smooth(rd["df"]["done"].astype(float), win)
        ax.plot(rd["df"]["steps"] / 1e6, rate, label=m["label"],
                color=m["color"], linestyle=m["ls"], lw=1.8)
    _apply_defaults(ax, "Episode completion rate: all runs",
                    "Steps (M)", f"Completion rate (rolling {win}-ep)", pct_y=True)
    fig.tight_layout()
    _save(fig, out_dir, "completion_all_runs")


def fig_stage_progression(run_data: list[dict], out_dir: Path) -> None:
    stage_colors = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
    stage_labels = ["Stage 1 (1 item)", "Stage 2 (2 items)",
                    "Stage 3 (4 items)", "Stage 4 (all items)"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    bar_h = 0.5

    for yi, rd in enumerate(run_data):
        df, meta = rd["df"], rd["meta"]
        trans = stage_first_steps(df)
        max_step = df["steps"].max()
        stages = sorted(trans.keys())
        for i, stage in enumerate(stages):
            start = trans[stage]
            end = trans[stages[i + 1]] if i + 1 < len(stages) else max_step
            ax.barh(yi, (end - start) / 1e6, left=start / 1e6, height=bar_h,
                    color=stage_colors[stage - 1], edgecolor="white", lw=0.5)

    ax.set_yticks(range(len(run_data)))
    ax.set_yticklabels([rd["meta"]["label"] for rd in run_data])
    ax.set_xlabel("Environment steps (M)")
    ax.set_title("Curriculum stage progression")
    ax.legend(handles=[Patch(facecolor=stage_colors[i], label=stage_labels[i])
                        for i in range(4)], loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "stage_progression")


def fig_overview_combined(run_data: list[dict], out_dir: Path, win: int) -> None:
    """Stack the three all-run line charts (reward, delivery, completion) and the
    curriculum stage progression into a single shared-x figure."""
    fig, axes = plt.subplots(
        4, 1, figsize=(10, 13), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 3, 3, 2], hspace=0.18),
    )
    ax_r, ax_d, ax_c, ax_s = axes

    # Panel 1: reward
    for rd in run_data:
        m = rd["meta"]
        ax_r.plot(rd["df"]["steps"] / 1e6, smooth(rd["df"]["avg50"], win),
                  label=m["label"], color=m["color"], linestyle=m["ls"], lw=1.8)
    ax_r.set_title("Training overview: all runs")
    ax_r.set_ylabel(f"Avg-50 reward\n(smoothed, win={win})")
    ax_r.legend(fontsize=8, ncol=2)
    ax_r.grid(True, alpha=0.3)

    # Panel 2: delivery fraction (each run uses its own goal's stage map)
    for rd in run_data:
        m = rd["meta"]
        ax_d.plot(rd["df"]["steps"] / 1e6, delivery_frac(rd["df"], m["goal"], win),
                  label=m["label"], color=m["color"], linestyle=m["ls"], lw=1.8)
    ax_d.set_ylabel("Delivered /\nactive items")
    ax_d.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax_d.grid(True, alpha=0.3)

    # Panel 3: completion rate
    for rd in run_data:
        m = rd["meta"]
        ax_c.plot(rd["df"]["steps"] / 1e6, smooth(rd["df"]["done"].astype(float), win),
                  label=m["label"], color=m["color"], linestyle=m["ls"], lw=1.8)
    ax_c.set_ylabel(f"Completion rate\n(rolling {win}-ep)")
    ax_c.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax_c.grid(True, alpha=0.3)

    # Panel 4: curriculum stage progression
    stage_colors = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
    stage_labels = ["Stage 1 (1 item)", "Stage 2 (2 items)",
                    "Stage 3 (4 items)", "Stage 4 (all items)"]
    bar_h = 0.5
    for yi, rd in enumerate(run_data):
        df = rd["df"]
        trans = stage_first_steps(df)
        max_step = df["steps"].max()
        stages = sorted(trans.keys())
        for i, stage in enumerate(stages):
            start = trans[stage]
            end = trans[stages[i + 1]] if i + 1 < len(stages) else max_step
            ax_s.barh(yi, (end - start) / 1e6, left=start / 1e6, height=bar_h,
                      color=stage_colors[stage - 1], edgecolor="white", lw=0.5)
    ax_s.set_yticks(range(len(run_data)))
    ax_s.set_yticklabels([rd["meta"]["label"] for rd in run_data])
    ax_s.set_ylabel("Curriculum\nstage")
    ax_s.set_xlabel("Environment steps (M)")
    ax_s.legend(handles=[Patch(facecolor=stage_colors[i], label=stage_labels[i])
                         for i in range(4)], loc="lower right", fontsize=8, ncol=2)
    ax_s.grid(True, axis="x", alpha=0.3)

    _save(fig, out_dir, "overview_combined")


def fig_sq1(run_data: list[dict], out_dir: Path, win: int) -> None:
    targets = {"run1_ont_food_prox5", "run2_vanilla_food"}
    sq1 = [rd for rd in run_data if rd["meta"]["name"] in targets]
    if len(sq1) < 2:
        print("  [skip] SQ1 plot: need run1 + run2")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for rd in sq1:
        m = rd["meta"]
        df = rd["df"]
        axes[0].plot(df["steps"] / 1e6, smooth(df["avg50"], win),
                     label=m["label"], color=m["color"], linestyle=m["ls"], lw=2)
        axes[1].plot(df["steps"] / 1e6, delivery_frac(df, "collect_food", win),
                     label=m["label"], color=m["color"], linestyle=m["ls"], lw=2)

    _apply_defaults(axes[0], "SQ1: Reward", "Steps (M)", "Avg-50 reward")
    _apply_defaults(axes[1], "SQ1: Delivery fraction", "Steps (M)",
                    "Delivered / active items", pct_y=True)
    fig.suptitle("SQ1: Ontology-Guided vs Vanilla Baseline (collect_food)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "sq1_ont_vs_vanilla")


def fig_sq3(run_data: list[dict], out_dir: Path, win: int) -> None:
    targets = {"run1_ont_food_prox5", "run3_ont_food_prox0"}
    sq3 = [rd for rd in run_data if rd["meta"]["name"] in targets]
    if len(sq3) < 2:
        print("  [skip] SQ3 plot: need run1 + run3")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for rd in sq3:
        m = rd["meta"]
        df = rd["df"]
        axes[0].plot(df["steps"] / 1e6, smooth(df["avg50"], win),
                     label=m["label"], color=m["color"], linestyle=m["ls"], lw=2)
        axes[1].plot(df["steps"] / 1e6, delivery_frac(df, "collect_food", win),
                     label=m["label"], color=m["color"], linestyle=m["ls"], lw=2)

    _apply_defaults(axes[0], "SQ3: Reward", "Steps (M)", "Avg-50 reward")
    _apply_defaults(axes[1], "SQ3: Delivery fraction", "Steps (M)",
                    "Delivered / active items", pct_y=True)
    fig.suptitle("SQ3: Shared ABox (prox=5) vs Independent ABox (prox=0)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "sq3_shared_vs_independent")


def fig_sq5(run_data: list[dict], out_dir: Path, win: int) -> None:
    targets = {"run1_ont_food_prox5", "run4_ont_trash_prox5"}
    sq5 = [rd for rd in run_data if rd["meta"]["name"] in targets]
    if not any(rd["meta"]["name"] == "run4_ont_trash_prox5" for rd in sq5):
        print("  [skip] SQ5 plot: need run4")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for rd in sq5:
        m = rd["meta"]
        df = rd["df"]
        axes[0].plot(df["steps"] / 1e6, smooth(df["avg50"], win),
                     label=m["label"], color=m["color"], linestyle=m["ls"], lw=2)
        frac = delivery_frac(df, m["goal"], win)
        axes[1].plot(df["steps"] / 1e6, frac,
                     label=m["label"], color=m["color"], linestyle=m["ls"], lw=2)

    _apply_defaults(axes[0], "SQ5: Reward", "Steps (M)", "Avg-50 reward")
    _apply_defaults(axes[1], "SQ5: Delivery fraction", "Steps (M)",
                    "Delivered / active items", pct_y=True)
    axes[1].annotate("Note: tasks differ\n(food vs trash)",
                     xy=(0.02, 0.95), xycoords="axes fraction",
                     fontsize=8, va="top", color="grey")
    fig.suptitle("SQ5: Framework Generalisation, Food vs Trash",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "sq5_generalisation")


# Summary table
def print_summary(run_data: list[dict]) -> None:
    print("\nStage transition summary (first step reaching each stage)")
    col = 28
    header = f"{'Run':<{col}} {'S1':>10} {'S2':>10} {'S3':>10} {'S4':>10}  {'Total eps':>10}  {'Total steps':>12}"
    print(header)
    print("-" * len(header))
    for rd in run_data:
        df, meta = rd["df"], rd["meta"]
        t = stage_first_steps(df)
        row = f"{meta['label']:<{col}}"
        for s in [1, 2, 3, 4]:
            v = t.get(s)
            row += f" {(f'{v/1e6:.1f}M' if v is not None else '-'):>10}"
        row += f"  {len(df):>10,}  {df['steps'].max():>12,}"
        print(row)

    print("\nFinal-100-episode averages")
    header2 = f"{'Run':<{col}} {'Reward':>10} {'Delivery%':>11} {'Complete%':>11}"
    print(header2)
    print("-" * len(header2))
    for rd in run_data:
        df, meta = rd["df"], rd["meta"]
        tail = df.tail(100)
        avg_r = tail["avg50"].mean()
        stage_map = STAGE_ITEMS[meta["goal"]]
        n_active = tail["stage"].map(stage_map).fillna(1)
        avg_d = (tail["balls"] / n_active).mean()
        avg_c = tail["done"].mean()
        print(f"{meta['label']:<{col}} {avg_r:>10.1f} {avg_d:>10.1%} {avg_c:>10.1%}")
    print()


# Entry point
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_root", type=Path, default=_DEFAULT_CKPT_ROOT,
                    help=f"Directory containing checkpoints_* subdirs "
                         f"(default: {_DEFAULT_CKPT_ROOT})")
    ap.add_argument("--out_dir", type=Path, default=_DEFAULT_OUT_DIR,
                    help=f"Output directory for figures (default: {_DEFAULT_OUT_DIR})")
    ap.add_argument("--smooth", type=int, default=_DEFAULT_SMOOTH,
                    help=f"Rolling-average window in episodes (default: {_DEFAULT_SMOOTH})")
    ap.add_argument("--max_steps", type=int, default=20_000_000,
                    help="Clip every run to this many env steps for a matched-budget "
                         "comparison (default: 20M). Use 0 to disable.")
    return ap.parse_args()


def main():
    args = parse_args()

    max_steps = args.max_steps if args.max_steps > 0 else None
    print(f"Loading logs from: {args.ckpt_root.resolve()}"
          + (f"  (clipped to {max_steps/1e6:.0f}M steps)" if max_steps else ""))
    run_data = []
    for meta in RUNS_META:
        df = load_run(args.ckpt_root, meta["name"], max_steps)
        if df is not None:
            run_data.append({"df": df, "meta": meta})
            print(f"  {meta['name']}: {len(df):,} eps, "
                  f"{df['steps'].max() / 1e6:.1f}M steps, "
                  f"stages seen: {sorted(df['stage'].unique())}")

    if not run_data:
        sys.exit("No training logs found. Check --ckpt_root.")

    print_summary(run_data)

    print(f"Saving figures to: {args.out_dir.resolve()}/")
    fig_reward_all(run_data, args.out_dir, args.smooth)
    fig_delivery_all(run_data, args.out_dir, args.smooth)
    fig_completion_all(run_data, args.out_dir, args.smooth)
    fig_stage_progression(run_data, args.out_dir)
    fig_overview_combined(run_data, args.out_dir, args.smooth)
    fig_sq1(run_data, args.out_dir, args.smooth)
    fig_sq3(run_data, args.out_dir, args.smooth)
    fig_sq5(run_data, args.out_dir, args.smooth)

    print(f"\nDone. {len(run_data)} run(s) plotted, 8 figures saved.")


if __name__ == "__main__":
    main()
