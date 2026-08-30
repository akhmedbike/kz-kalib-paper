"""Aggregate per-run metric JSONs into the paper's markdown tables.

    python -m kzcalib.aggregate

Produces results/tables/{summary,operating_points,routing,per_tag,bootstrap}.md
Neural rows are mean ± std over seeds; the CRF is a single released model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEEDS = (13, 42, 123)
NEURAL = ("kazbert", "kazroberta", "xlmr")
DISPLAY = {"kazbert": "KazBERT", "kazroberta": "KazRoBERTa",
           "xlmr": "XLM-R", "crf": "CRF"}


def load(root: Path, run: str) -> dict | None:
    p = root / "metrics" / f"{run}.json"
    return json.loads(p.read_text()) if p.exists() else None


def runs(model: str, root: Path) -> list[dict]:
    return [d for d in (load(root, f"{model}_{s}") for s in SEEDS) if d]


def ms(vals, scale=1.0, prec=2) -> str:
    v = np.array(vals, dtype=float) * scale
    if len(v) == 1:
        return f"{v[0]:.{prec}f}"
    return f"{v.mean():.{prec}f} ± {v.std(ddof=1):.{prec}f}"


def table_summary(root: Path) -> str:
    lines = [
        "# Table 3 (harmonised): tagging and calibration, one split, one code path",
        "",
        "Test split N = 1,594 tokens; temperature fitted on dev (N = 1,566): "
        "primary protocol NLL, legacy protocol ECE grid (0.8–5.0, step 0.1). "
        "Neural rows: mean ± std over 3 seeds. acc = token accuracy.",
        "",
        "| Model | acc (%) | macro F1 | T(NLL) | ECE raw (%) | ECE scaled NLL (%) "
        "| ECE scaled grid (%) | Brier full | Brier top-1 | NLL | AURC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in NEURAL:
        rs = runs(m, root)
        if not rs:
            continue
        row = [
            DISPLAY[m],
            ms([r["raw"]["accuracy"] for r in rs], 100),
            ms([r["raw"]["macro_f1"] for r in rs], 100),
            ms([r["T_nll"]["T"] for r in rs], 1, 2),
            ms([r["raw"]["ece"] for r in rs], 100),
            ms([r["scaled_nll"]["ece"] for r in rs], 100),
            ms([r["scaled_ece_grid"]["ece"] for r in rs], 100),
            ms([r["raw"]["brier"] for r in rs], 1, 4),
            ms([r["raw"]["brier_top1"] for r in rs], 1, 4),
            ms([r["raw"]["nll"] for r in rs], 1, 4),
            ms([r["raw"]["aurc"] for r in rs], 1, 4),
        ]
        lines.append("| " + " | ".join(row) + " |")
    crf = load(root, "crf")
    if crf:
        r = crf["raw"]
        lines.append(
            f"| CRF | {r['accuracy']*100:.2f} | {r['macro_f1']*100:.2f} | — "
            f"| {r['ece']*100:.2f} | — | — | {r['brier']:.4f} "
            f"| {r['brier_top1']:.4f} | {r['nll']:.4f} | {r['aurc']:.4f} |")
    lines.append("")
    lines.append("Accuracy is identical raw vs scaled (arg-max preserving).")
    return "\n".join(lines) + "\n"


def _ops_across(rs: list[dict], key: str, field: str, tau: float) -> list[float]:
    out = []
    for r in rs:
        op = next(o for o in r[key]["operating_points"] if o["tau"] == tau)
        if op[field] is not None:
            out.append(op[field])
    return out


def table_operating(root: Path) -> str:
    lines = [
        "# Operating points on the deployed thresholds",
        "",
        "Coverage = fraction auto-decided; risk = error among them. "
        "Neural rows: mean ± std over 3 seeds.",
        "",
        "| Model | conf source | cov@0.70 (%) | risk@0.70 (%) | cov@0.85 (%) | risk@0.85 (%) |",
        "|---|---|---|---|---|---|",
    ]
    for m in NEURAL:
        rs = runs(m, root)
        if not rs:
            continue
        for key, label in (("raw", "raw"), ("scaled_nll", "scaled (T by dev NLL)")):
            lines.append(
                f"| {DISPLAY[m]} | {label} "
                f"| {ms(_ops_across(rs, key, 'coverage', 0.70), 100)} "
                f"| {ms(_ops_across(rs, key, 'risk', 0.70), 100)} "
                f"| {ms(_ops_across(rs, key, 'coverage', 0.85), 100)} "
                f"| {ms(_ops_across(rs, key, 'risk', 0.85), 100)} |")
    crf = load(root, "crf")
    if crf:
        ops = {op["tau"]: op for op in crf["raw"]["operating_points"]}
        lines.append(
            f"| CRF | marginals | {ops[0.70]['coverage']*100:.2f} "
            f"| {ops[0.70]['risk']*100:.2f} | {ops[0.85]['coverage']*100:.2f} "
            f"| {ops[0.85]['risk']*100:.2f} |")
    return "\n".join(lines) + "\n"


def table_routing(root: Path) -> str:
    lines = [
        "# Routing shift under temperature scaling (NLL protocol)",
        "",
        "Labels never change; only the threshold decision does.",
        "",
        "| Model | τ | cov raw (%) | cov scaled (%) | Δcov (pp) | cross↓ n "
        "| err among↓ (%) | err among kept (%) | risk raw (%) | risk scaled (%) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in NEURAL:
        rs = runs(m, root)
        for tau in (0.70, 0.85):
            entries = [r["thresholds_nll"]["routing_shift"][f"tau={tau}"] for r in rs]
            down_n = [e["n_cross_down"] for e in entries]
            d_err = [e["down_error_rate"] for e in entries if e["down_error_rate"] is not None]
            k_err = [e["error_rate_among_always_above"] for e in entries
                     if e["error_rate_among_always_above"] is not None]
            lines.append(
                f"| {DISPLAY[m]} | {tau} | {np.mean([e['coverage_raw'] for e in entries])*100:.2f} "
                f"| {np.mean([e['coverage_scaled'] for e in entries])*100:.2f} "
                f"| {np.mean([e['coverage_delta_pp'] for e in entries]):+.2f} "
                f"| {np.mean(down_n):.1f} "
                f"| {np.mean(d_err)*100:.2f} "
                f"| {np.mean(k_err)*100:.2f} "
                f"| {np.mean([e['risk_raw'] for e in entries])*100:.2f} "
                f"| {np.mean([e['risk_scaled'] for e in entries])*100:.2f} |")
    return "\n".join(lines) + "\n"


def table_per_tag(root: Path) -> str:
    lines = [
        "# Per-tag calibration (descriptive; seed 42 neural runs + CRF)",
        "",
        "| UPOS | model | n | acc (%) | mean conf (%) | gap (pp) |",
        "|---|---|---|---|---|---|",
    ]
    sources = []
    for m in NEURAL:
        d = load(root, f"{m}_42")
        if d:
            sources.append((DISPLAY[m], d["raw"]["per_tag"]))
    crf = load(root, "crf")
    if crf:
        sources.append(("CRF", crf["raw"]["per_tag"]))
    tags = sorted({t for _, pt in sources for t in pt}, key=lambda t: -pt_n(sources, t))
    for t in tags:
        for name, pt in sources:
            e = pt.get(t)
            if not e or e["n"] == 0:
                continue
            lines.append(f"| {t} | {name} | {e['n']} | {e['acc']*100:.1f} "
                         f"| {e['mean_conf']*100:.1f} | {e['gap_pp']:+.1f} |")
    return "\n".join(lines) + "\n"


def pt_n(sources, t) -> int:
    for _, pt in sources:
        if t in pt:
            return pt[t]["n"]
    return 0


def _ci(b: dict, key: str, scale: float, prec: int) -> str:
    e = b.get(key)
    if not e:
        return "—"
    return (f"{e['ci95_low']*scale:.{prec}f}–{e['ci95_high']*scale:.{prec}f} "
            f"({e['mean']*scale:.{prec}f})")


def table_bootstrap(root: Path) -> str:
    lines = [
        "# Bootstrap 95% CIs (1,000 token resamples, test split, seed 42 runs)",
        "",
        "| Model | conf | ECE (%) | Brier top-1 | cov@0.85 (%) | risk@0.85 (%) |",
        "|---|---|---|---|---|---|",
    ]
    for m in NEURAL:
        r = load(root, f"{m}_42")
        if not r:
            continue
        for bk, label in (("bootstrap_raw", "raw"),
                          ("bootstrap_scaled_nll", "scaled (NLL)")):
            b = r[bk]
            lines.append(
                f"| {DISPLAY[m]} | {label} | {_ci(b, 'ece', 100, 2)} "
                f"| {_ci(b, 'brier_top1', 1, 4)} | {_ci(b, 'coverage@0.85', 100, 2)} "
                f"| {_ci(b, 'risk@0.85', 100, 2)} |")
    crf = load(root, "crf")
    if crf:
        b = crf["bootstrap_raw"]
        lines.append(f"| CRF | marginals | {_ci(b, 'ece', 100, 2)} "
                     f"| {_ci(b, 'brier_top1', 1, 4)} | {_ci(b, 'coverage@0.85', 100, 2)} "
                     f"| {_ci(b, 'risk@0.85', 100, 2)} |")
    lines.append("")
    lines.append("Full CI sets (all metrics) live in results/metrics/*.json "
                 "under bootstrap_raw / bootstrap_scaled_nll.")
    return "\n".join(lines) + "\n"


def main():
    root = Path(__file__).resolve().parent.parent / "results"
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("summary", table_summary),
                     ("operating_points", table_operating),
                     ("routing", table_routing),
                     ("per_tag", table_per_tag),
                     ("bootstrap", table_bootstrap)):
        path = out / f"{name}.md"
        path.write_text(fn(root), encoding="utf-8")
        print(f"wrote tables/{name}.md")


if __name__ == "__main__":
    main()
