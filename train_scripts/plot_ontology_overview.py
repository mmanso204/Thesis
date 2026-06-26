"""General overview diagram of the thesis ontology (class hierarchy only).

Top-down, ONE level deep: each root sits above its direct subclasses. The three
roots (Entity, State, Goal) are drawn as separate sections divided by dotted
lines.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# root -> direct subclasses (one depth only)
SECTIONS = [
    ("Entity", "#cfe8ff", "#1f6feb",
     ["Agent", "Team", "Cell", "Room", "Item"]),
    ("State", "#d8f5d0", "#2e9e3f",
     ["World", "Belief"]),
    ("Goal", "#ffe2c2", "#e8820c",
     ["ActiveGoal", "NextGoal", "ReachableGoal", "UnreachableGoal",
      "CompletedGoal", "SkippedGoal", "BlockedGoal", "Atomic", "GroupedGoal"]),
]

BW, BH = 1.7, 0.62          # box width / height
SECTION_H = 2.6             # vertical span per section
COLW = 2.0                  # horizontal spacing between child boxes

max_children = max(len(c) for _, _, _, c in SECTIONS)
fig_w = max(11, max_children * COLW + 1.5)
fig, ax = plt.subplots(figsize=(fig_w, len(SECTIONS) * SECTION_H + 0.6))


def box(cx, cy, label, face, edge, root=False):
    ax.add_patch(FancyBboxPatch(
        (cx - BW / 2, cy - BH / 2), BW, BH,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2.0 if root else 1.1,
        edgecolor=edge if root else "#5f6368",
        facecolor=face, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=11 if root else 8.5,
            fontweight="bold" if root else "normal", zorder=4)


total_w = max_children * COLW
for i, (root, face, edge, children) in enumerate(SECTIONS):
    top = -i * SECTION_H
    root_y = top - 0.45
    child_y = top - 1.7
    # root centered over the full width
    cx0 = total_w / 2
    box(cx0, root_y, root, face, edge, root=True)
    # children spread evenly, centered
    span = (len(children) - 1) * COLW
    start = cx0 - span / 2
    for j, ch in enumerate(children):
        cx = start + j * COLW
        ax.add_line(Line2D([cx0, cx], [root_y - BH / 2, child_y + BH / 2],
                           color=edge, lw=1.0, alpha=0.7, zorder=1))
        box(cx, child_y, ch, face, edge)
    # dotted divider below this section (except after the last)
    if i < len(SECTIONS) - 1:
        ydiv = top - SECTION_H + 0.35
        ax.add_line(Line2D([-0.5, total_w + 0.5], [ydiv, ydiv],
                           color="#888888", lw=1.2, ls=(0, (4, 4)), zorder=0))

ax.set_xlim(-1.0, total_w + 1.0)
ax.set_ylim(-len(SECTIONS) * SECTION_H + 0.2, 0.6)
ax.axis("off")
ax.set_title("Ontology class hierarchy: general overview", fontsize=15,
             fontweight="bold", pad=10)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
out = "results/ontology_overview.png"
plt.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)
