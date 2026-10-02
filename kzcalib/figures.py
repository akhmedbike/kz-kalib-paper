"""All paper figures, generated from results/metrics/*.json.

Run after kzcalib.analyze has produced per-run metric files:

    python -m kzcalib.figures

Outputs PNG + PDF (300 dpi) into results/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODELS = ["kazbert", "kazroberta", "xlmr", "crf"]
DISPLAY = {
    "kazbert": "KazBERT",
    "kazroberta": "KazRoBERTa",
    "xlmr": "XLM-R",
    "crf": "CRF",
}
COLORS = {
    "kazbert": "#0072B2",
    "kazroberta": "#E69F00",
    "xlmr": "#009E73",
    "crf": "#CC79A7",
}
TAUS = (0.70, 0.85)


def load_metrics(root: Path, run: str) -> dict:
    return json.loads((root / "metrics" / f"{run}.json").read_text())


def runs_for(model: str, root: Path) -> list[str]:
    seeds = sorted(p.stem.split("_")[1] for p in (root / "metrics").glob(f"{model}_*.json"))
    return [f"{model}_{s}" for s in seeds]


def _save(fig, root: Path, name: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(root / "figures" / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.png|.pdf|.svg")


# ---------------------------------------------------------------------------


def fig_reliability(root: Path) -> None:
    """Reliability diagrams, one panel per model (raw; neural also scaled)."""
    panels = []
    for m in ("kazbert", "kazroberta", "xlmr"):
        run = runs_for(m, root)[0] if runs_for(m, root) else None
        if run:
            panels.append((f"{DISPLAY[m]} (raw)", load_metrics(root, run)["raw"]))
            panels.append((f"{DISPLAY[m]} (scaled)", load_metrics(root, run)["scaled_nll"]))
    if (root / "metrics" / "crf.json").exists():
        panels.append((f"{DISPLAY['crf']} (marginals)", load_metrics(root, "crf")["raw"]))
        panels.append((f"{DISPLAY['crf']} (scaled)", load_metrics(root, "crf")["scaled_probs"]))

    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows),
                             squeeze=False, layout="constrained")
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (title, rep) in zip(axes.flat, panels):
        bins = rep["reliability_bins"]
        centers = [b["center"] for b in bins]
        accs = [b["accuracy"] if b["accuracy"] is not None else 0 for b in bins]
        counts = np.array([b["count"] for b in bins], dtype=float)
        width = 0.09
        ax.bar(centers, accs, width=width * 1.8, color="#56B4E9",
               edgecolor="none", label="accuracy")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="perfect")
        ax2 = ax.twinx()
        ax2.bar(centers, counts / counts.sum(), width=width, color="black",
                alpha=0.35, label="share")
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes.flat[0].legend(fontsize=6, loc="upper left")
    _save(fig, root, "fig_reliability_all")


def fig_risk_coverage(root: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for m in MODELS:
        if m == "crf":
            if not (root / "metrics" / "crf.json").exists():
                continue
            rep = load_metrics(root, "crf")["raw"]
            ax.plot(rep["risk_coverage"]["coverage"],
                    rep["risk_coverage"]["risk"],
                    color=COLORS[m], label=DISPLAY[m], lw=1.6)
            continue
        runs = runs_for(m, root)
        if not runs:
            continue
        curves = [load_metrics(root, r)["raw"]["risk_coverage"] for r in runs]
        cov = np.array(curves[0]["coverage"])
        risks = np.array([c["risk"] for c in curves])
        ax.plot(cov, risks.mean(axis=0), color=COLORS[m], label=DISPLAY[m], lw=1.6)
        ax.fill_between(cov, risks.min(axis=0), risks.max(axis=0),
                        color=COLORS[m], alpha=0.15, lw=0)
    for tau in TAUS:
        ax.axvline(1 - tau, color="gray", lw=0.7, ls=":")
        ax.text(1 - tau, ax.get_ylim()[1] * 0.92, f"τ={tau}", fontsize=7,
                ha="center", color="gray")
    ax.set_xlabel("coverage (fraction of predictions retained)")
    ax.set_ylabel("selective risk")
    ax.set_xlim(0.4, 1.0)
    ax.legend(fontsize=7)
    _save(fig, root, "fig_risk_coverage_all")


def fig_ece_scaling(root: Path) -> None:
    """Raw vs temperature-scaled ECE (NLL protocol), mean ± SD over seeds."""
    models = [m for m in ("kazbert", "kazroberta", "xlmr") if runs_for(m, root)]
    raw = [[load_metrics(root, r)["raw"]["ece"] * 100 for r in runs_for(m, root)]
           for m in models]
    scl = [[load_metrics(root, r)["scaled_nll"]["ece"] * 100 for r in runs_for(m, root)]
           for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for off, vals, color, label in ((-0.16, raw, "#999999", "raw"),
                                    (+0.16, scl, "#D55E00", "scaled (T by dev NLL)")):
        means = [np.mean(v) for v in vals]
        errs = [np.std(v, ddof=1) for v in vals]
        ax.bar(x + off, means, width=0.3, color=color, label=label,
               yerr=errs, capsize=3, error_kw={"lw": 0.9})
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models])
    ax.set_ylabel("ECE (%)")  # error bars: ±1 SD over the three seeds
    ax.legend(fontsize=7)
    _save(fig, root, "fig_ece_scaling")


def fig_f1_vs_ece(root: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for m in ("kazbert", "kazroberta", "xlmr", "crf"):
        if m == "crf":
            if not (root / "metrics" / "crf.json").exists():
                continue
            rep = load_metrics(root, "crf")["raw"]
            ax.scatter(rep["ece"] * 100, rep["accuracy"] * 100,
                       color=COLORS[m], label=DISPLAY[m], s=45, zorder=3)
            continue
        runs = runs_for(m, root)
        if not runs:
            continue
        f1s = [load_metrics(root, r)["raw"]["accuracy"] * 100 for r in runs]
        eces = [load_metrics(root, r)["raw"]["ece"] * 100 for r in runs]
        ax.scatter(eces, f1s, color=COLORS[m], s=28, alpha=0.7, zorder=3)
        ax.scatter([np.mean(eces)], [np.mean(f1s)], color=COLORS[m],
                   label=DISPLAY[m], s=70, marker="X", zorder=4,
                   edgecolor="white", lw=0.6)
    ax.set_xlabel("raw ECE (%)")
    ax.set_ylabel("token accuracy (%)")
    ax.legend(fontsize=7)
    _save(fig, root, "fig_f1_vs_ece")


def fig_routing_shift(root: Path) -> None:
    """Tokens crossing each threshold after scaling, with error composition."""
    models = [m for m in ("kazbert", "kazroberta", "xlmr") if runs_for(m, root)]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, tau in zip(axes, TAUS):
        labels, down_rates, keep_rates = [], [], []
        for m in models:
            rs = [load_metrics(root, r)["thresholds_nll"]["routing_shift"][f"tau={tau}"]
                  for r in runs_for(m, root)]
            d = np.mean([x["down_error_rate"] for x in rs if x["down_error_rate"] is not None])
            k = np.mean([x["error_rate_among_always_above"]
                         for x in rs if x["error_rate_among_always_above"] is not None])
            labels.append(DISPLAY[m])
            down_rates.append(d * 100)
            keep_rates.append(k * 100)
        x = np.arange(len(labels))
        ax.bar(x - 0.17, down_rates, width=0.32, color="#D55E00",
               label="tokens losing auto status")
        ax.bar(x + 0.17, keep_rates, width=0.32, color="#0072B2",
               label="tokens keeping auto status")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"τ = {tau}", fontsize=9)
        ax.set_ylabel("error rate (%)")
    axes[0].legend(fontsize=7)
    _save(fig, root, "fig_routing_shift")


def main():
    root = Path(__file__).resolve().parent.parent / "results"
    (root / "figures").mkdir(parents=True, exist_ok=True)
    fig_reliability(root)
    fig_risk_coverage(root)
    fig_ece_scaling(root)
    fig_f1_vs_ece(root)
    fig_routing_shift(root)


if __name__ == "__main__":
    main()
