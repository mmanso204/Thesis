#!/usr/bin/env python3
"""
Analyse and plot the evaluation results (50 stochastic episodes per run) and
run significance tests on the key thesis contrasts.

Reads the per-episode tables in results/eval/run*.txt, then:
  1. Plots per-episode reward (box + strip) and per-episode delivery for all runs.
  2. Runs significance tests on the pre-registered contrasts:
       SQ1  Ontology vs Vanilla        : run1 vs run2,  run3 vs run2
       SQ3  Shared vs Independent ABox  : run1 vs run3
     Reward  -> Mann-Whitney U (rank-based, robust to the multimodal, non-normal
                reward distribution) + Welch's t-test + Cliff's delta effect size.
     Delivery-> Mann-Whitney U on per-episode items delivered.

Outputs:
    results/figures/eval_reward_box.{pdf,png}
    results/figures/eval_delivery_bar.{pdf,png}
    results/eval_significance.txt   (full report, also printed to stdout)

Usage:
    python train_scripts/analyze_eval.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Larger fonts for thesis figures (legible at print size).
plt.rcParams.update({
    "font.size":        16,
    "axes.titlesize":   17,
    "axes.labelsize":   17,
    "xtick.labelsize":  15,
    "ytick.labelsize":  15,
    "legend.fontsize":  15,
})

# Run definitions (colours/labels match plot_results.py)
RUNS_META = [
    dict(name="run1_ont_food_prox5", label="Ont + Shared\n(run1)",  short="run1", color="#1f77b4"),
    dict(name="run2_vanilla_food",   label="Vanilla\n(run2)",        short="run2", color="#ff7f0e"),
    dict(name="run3_ont_food_prox0", label="Ont + Indep\n(run3)",    short="run3", color="#2ca02c"),
    dict(name="run4_ont_trash_prox5", label="Ont + Trash\n(run4)",   short="run4", color="#9467bd"),
]

_HERE = Path(__file__).parent
_REPO = _HERE.parent
_EVAL_DIR = _REPO / "results" / "eval"
_OUT_DIR = _REPO / "results" / "figures"
_REPORT = _REPO / "results" / "eval_significance.txt"

# Matches e.g.  "   3      974.0      3/4           ✗    4000"
_ROW_RE = re.compile(r"^\s*\d+\s+(-?\d+\.\d+)\s+(\d+)/(\d+)\s")


def parse_eval(path: Path):
    """Return (rewards, delivered, n_items) arrays from one eval .txt."""
    rewards, delivered, n_items = [], [], None
    for line in path.read_text().splitlines():
        m = _ROW_RE.match(line)
        if m:
            rewards.append(float(m.group(1)))
            delivered.append(int(m.group(2)))
            n_items = int(m.group(3))
    return np.array(rewards), np.array(delivered, dtype=float), n_items


def cliffs_delta(a, b):
    """Cliff's delta effect size: P(a>b) - P(a<b). |d|: .147 small, .33 med, .474 large."""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def contrast(name, a, b, lines, metric="reward"):
    """Run the test battery for one A-vs-B contrast and append to `lines`."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    # Mann-Whitney U (two-sided, rank-based)
    u, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Welch t-test (unequal variance)
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    delta = cliffs_delta(a, b)
    d = cohens_d(a, b)

    def stars(p):
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

    lines.append(f"  {name}")
    lines.append(f"    {'A':<6} mean={a.mean():8.2f}  sd={a.std(ddof=1):7.2f}  median={np.median(a):8.2f}  n={len(a)}")
    lines.append(f"    {'B':<6} mean={b.mean():8.2f}  sd={b.std(ddof=1):7.2f}  median={np.median(b):8.2f}  n={len(b)}")
    lines.append(f"    diff (A-B) = {a.mean()-b.mean():+.2f}")
    lines.append(f"    Mann-Whitney U = {u:.0f}   p = {p_mw:.3e}  {stars(p_mw)}")
    lines.append(f"    Welch t({_welch_df(a,b):.1f}) = {t:+.2f}   p = {p_t:.3e}  {stars(p_t)}")
    lines.append(f"    Cliff's delta = {delta:+.3f}  ({_mag(abs(delta), (.147,.33,.474))})")
    lines.append(f"    Cohen's d     = {d:+.3f}  ({_mag(abs(d), (.2,.5,.8))})")
    lines.append("")
    return p_mw


def _welch_df(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))


def _mag(x, thr):
    s, m, l = thr
    return "large" if x >= l else "medium" if x >= m else "small" if x >= s else "negligible"


def main():
    data = {}
    for meta in RUNS_META:
        path = _EVAL_DIR / f"{meta['name']}.txt"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        rew, deliv, n_items = parse_eval(path)
        data[meta["name"]] = dict(reward=rew, delivered=deliv, n_items=n_items, **meta)
        print(f"  {meta['short']}: {len(rew)} eps, reward mean={rew.mean():.1f}, "
              f"delivered mean={deliv.mean():.2f}/{n_items}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: per-episode reward (box + jittered points)
    order = [m["name"] for m in RUNS_META if m["name"] in data]
    rewards = [data[n]["reward"] for n in order]
    colors = [data[n]["color"] for n in order]
    labels = [data[n]["label"] for n in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(rewards, patch_artist=True, widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    for med in bp["medians"]:
        med.set_color("black")
    rng = np.random.default_rng(0)
    for i, (r, c) in enumerate(zip(rewards, colors), start=1):
        x = rng.normal(i, 0.06, size=len(r))
        ax.scatter(x, r, s=14, color=c, alpha=0.6, edgecolor="none", zorder=3)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Episode reward")
    ax.set_title("Evaluation reward over 50 stochastic episodes\n(Stage 2, 4 items)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_OUT_DIR / f"eval_reward_box.{ext}", dpi=150)
    plt.close(fig)

    # Figure 2: mean items delivered (bar + 95% CI)
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [data[n]["delivered"].mean() for n in order]
    cis = []
    for n in order:
        d = data[n]["delivered"]
        sem = d.std(ddof=1) / np.sqrt(len(d))
        cis.append(1.96 * sem)
    bars = ax.bar(range(len(order)), means, yerr=cis, capsize=5,
                  color=colors, alpha=0.7, edgecolor="black")
    for i, n in enumerate(order):
        ax.text(i, means[i] + cis[i] + 0.02, f"{means[i]:.2f}",
                ha="center", va="bottom", fontsize=15)
    n_items = data[order[0]]["n_items"]
    ax.axhline(n_items, ls="--", color="gray", lw=1, label=f"all {n_items} items")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean items delivered")
    ax.set_ylim(0, n_items + 0.3)
    ax.set_title("Mean items delivered per episode (±95% CI)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_OUT_DIR / f"eval_delivery_bar.{ext}", dpi=150)
    plt.close(fig)

    # Significance tests
    L = []
    L.append("=" * 72)
    L.append("EVALUATION SIGNIFICANCE REPORT  (50 stochastic episodes per run)")
    L.append("=" * 72)
    L.append("Tests: Mann-Whitney U (primary, rank-based) + Welch t-test.")
    L.append("Effect size: Cliff's delta (rank) and Cohen's d.")
    L.append("Significance: *** p<.001  ** p<.01  * p<.05  ns not significant")
    L.append("In each contrast, A is the first-named run, B the second.")
    L.append("")

    def have(*names):
        return all(n in data for n in names)

    L.append("-" * 72)
    L.append("SQ1 - Does the ontology help? (vs vanilla baseline run2)")
    L.append("-" * 72)
    L.append(" REWARD:")
    if have("run1_ont_food_prox5", "run2_vanilla_food"):
        contrast("run1 (Ont+Shared)  vs  run2 (Vanilla)",
                 data["run1_ont_food_prox5"]["reward"],
                 data["run2_vanilla_food"]["reward"], L)
    if have("run3_ont_food_prox0", "run2_vanilla_food"):
        contrast("run3 (Ont+Indep)   vs  run2 (Vanilla)",
                 data["run3_ont_food_prox0"]["reward"],
                 data["run2_vanilla_food"]["reward"], L)
    L.append(" DELIVERY (items/episode):")
    if have("run1_ont_food_prox5", "run2_vanilla_food"):
        contrast("run1 (Ont+Shared)  vs  run2 (Vanilla)",
                 data["run1_ont_food_prox5"]["delivered"],
                 data["run2_vanilla_food"]["delivered"], L, metric="delivery")
    if have("run3_ont_food_prox0", "run2_vanilla_food"):
        contrast("run3 (Ont+Indep)   vs  run2 (Vanilla)",
                 data["run3_ont_food_prox0"]["delivered"],
                 data["run2_vanilla_food"]["delivered"], L, metric="delivery")

    L.append("-" * 72)
    L.append("SQ3 - Shared vs Independent ABox (run1 prox5 vs run3 prox0)")
    L.append("-" * 72)
    L.append(" REWARD:")
    if have("run1_ont_food_prox5", "run3_ont_food_prox0"):
        contrast("run1 (Shared)  vs  run3 (Indep)",
                 data["run1_ont_food_prox5"]["reward"],
                 data["run3_ont_food_prox0"]["reward"], L)
    L.append(" DELIVERY (items/episode):")
    if have("run1_ont_food_prox5", "run3_ont_food_prox0"):
        contrast("run1 (Shared)  vs  run3 (Indep)",
                 data["run1_ont_food_prox5"]["delivered"],
                 data["run3_ont_food_prox0"]["delivered"], L, metric="delivery")

    L.append("=" * 72)
    L.append("NOTE: All 4 runs are single training seeds; these tests compare")
    L.append("episode-level variability within a fixed policy, not across seeds.")
    L.append("They show whether one policy reliably out-delivers another at eval")
    L.append("time, but cannot rule out training-seed variance. State this caveat")
    L.append("in the thesis.")
    L.append("=" * 72)

    report = "\n".join(L)
    print("\n" + report)
    _REPORT.write_text(report + "\n")
    print(f"\nSaved report  -> {_REPORT}")
    print(f"Saved figures -> {_OUT_DIR}/eval_reward_box.png, eval_delivery_bar.png")


if __name__ == "__main__":
    main()
