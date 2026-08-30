#!/usr/bin/env python3
"""Figure 1: confidence-aware evaluation pipeline (replaces the asset lost
with the original docx extraction; kept as code so it ships with the repo)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

BLUE, ORANGE, GREEN, PINK, GREY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#666666"

fig, ax = plt.subplots(figsize=(10.2, 4.6), layout="constrained")
ax.set_xlim(0, 10.2)
ax.set_ylim(0, 4.6)
ax.axis("off")


def box(x, y, w, h, title, lines, edge, fill, tsize=8.5, lsize=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fill, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color=edge)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - i * 0.30, ln, ha="center",
                va="center", fontsize=lsize, color="#222222")


def arrow(x1, y1, x2, y2, color=GREY, style="-", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, linestyle=style,
                                 shrinkA=2, shrinkB=2))


# Column 1 — taggers
box(0.15, 1.45, 1.95, 2.55, "Taggers",
    ["KazBERT · KazRoBERTa · XLM-R", "→ logits $z_i$",
     "", "Feature-based CRF", "→ marginals $p_i$"],
    BLUE, "#E8F1FA")
ax.text(1.12, 1.15, "$\\hat{y}_i = \\arg\\max_k z_{ik}$  (Eq. 1)",
        ha="center", fontsize=7.4, color=BLUE)

# Column 2 — confidence + temperature
box(2.75, 2.55, 2.10, 1.45, "Confidence",
    ["$c_i = \\max_k p_{ik}$", "", "Temperature $T$", "(fitted on dev, Eq. 5)"],
    ORANGE, "#FDF3E0")
box(2.75, 1.05, 2.10, 1.05, "",
    ["arg-max unchanged under T:", "labels fixed;", "confidence moves"],
    ORANGE, "#FDF3E0", tsize=1, lsize=7.4)

# Lane A — measurement
box(5.55, 3.05, 2.30, 1.25, "Measurement",
    ["ECE (Eq. 3) · Brier/NLL (Eq. 4)", "reliability diagrams",
     "AURC (Eq. 7)"],
    GREEN, "#E9F5F0")

# Lane B — decision layer
box(5.55, 0.55, 2.30, 2.05, "Decision layer",
    ["$\\tau = 0.85$ → auto-correct", "$\\tau = 0.70$ → review",
     "", "coverage / selective risk (Eq. 6)", "routing shift under $T$"],
    PINK, "#FAEDF4")

# Right note — what is claimed where
box(8.25, 0.55, 1.80, 3.75, "Claims",
    ["calibration error", "measured (4.1–4.3)", "",
     "threshold decisions", "measured (4.5–4.6)", "",
     "correction outcomes", "out of scope", "(Secs. 4.6, 6)"],
    GREY, "#F2F2F2", tsize=8.5, lsize=7.4)

# Arrows
arrow(2.10, 2.95, 2.75, 3.15)                    # taggers -> confidence
arrow(2.10, 2.30, 2.75, 1.75)                    # taggers -> T-note (labels)
arrow(4.85, 3.55, 5.55, 3.60, GREEN)             # confidence -> measurement
arrow(4.60, 2.55, 5.55, 1.55, PINK)              # confidence -> decision
arrow(3.80, 2.55, 3.80, 2.10, ORANGE, style="--")  # T note link
arrow(7.85, 3.55, 8.25, 3.30, GREY, style=":")
arrow(7.85, 1.60, 8.25, 1.50, GREY, style=":")

# Lane labels
ax.text(4.85, 4.25, "probs + labels", fontsize=7.2, color=GREEN, ha="center")
ax.text(4.85, 0.30, "calibrated $c_i$ only", fontsize=7.2, color=PINK, ha="center")

root = Path(__file__).resolve().parent.parent
for d in (root / "docs" / "figures", root / "results" / "figures"):
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "fig1_pipeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(d / "fig1_pipeline.pdf", bbox_inches="tight")
    fig.savefig(d / "fig1_pipeline.svg", bbox_inches="tight")
print("wrote docs/figures/fig1_pipeline.png|.pdf|.svg and results/figures/")
