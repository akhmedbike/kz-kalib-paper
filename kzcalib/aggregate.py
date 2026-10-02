"""Aggregate per-run metric JSONs into the paper's markdown tables.

    python -m kzcalib.aggregate

Produces results/tables/{summary,post_ts,operating_points,routing,per_tag,
bootstrap_sentence,bootstrap_token,ranking}.md
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


def _fmt_pair(a: float, b: float, scale=1.0, prec=2) -> str:
    return f"{a*scale:.{prec}f} → {b*scale:.{prec}f}"


def table_summary(root: Path) -> str:
    lines = [
        "# Table 3 (harmonised): tagging and raw calibration, one split, one code path",
        "",
        "Test split N = 1,594 tokens. Neural rows: mean ± std over 3 seeds. "
        "acc = token accuracy; aECE = equal-mass adaptive ECE.",
        "",
        "| Model | acc (%) | macro F1 | ECE raw (%) | aECE raw (%) "
        "| Brier full | Brier top-1 | NLL | AURC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in NEURAL:
        rs = runs(m, root)
        if not rs:
            continue
        row = [
            DISPLAY[m],
            ms([r["raw"]["accuracy"] for r in rs], 100),
            ms([r["raw"]["macro_f1"] for r in rs], 100),
            ms([r["raw"]["ece"] for r in rs], 100),
            ms([r["raw"]["adaptive_ece"] for r in rs], 100),
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
            f"| CRF | {r['accuracy']*100:.2f} | {r['macro_f1']*100:.2f} "
            f"| {r['ece']*100:.2f} | {r['adaptive_ece']*100:.2f} "
            f"| {r['brier']:.4f} | {r['brier_top1']:.4f} "
            f"| {r['nll']:.4f} | {r['aurc']:.4f} |")
    lines.append("")
    lines.append("Accuracy is identical raw vs scaled (arg-max preserving).")
    return "\n".join(lines) + "\n"


def table_post_ts(root: Path) -> str:
    """Proper scoring rules before/after temperature scaling (R1-5 / R2-4)."""
    lines = [
        "# Proper scoring rules, raw → temperature-scaled (test split)",
        "",
        "T fitted on development data by NLL (transformers: logits; CRF: "
        "marginals). Neural rows: seed means. Accuracy is unchanged by scaling.",
        "",
        "| Model | T | ECE (%) | aECE (%) | Brier full | Brier top-1 | NLL |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in NEURAL:
        rs = runs(m, root)
        if not rs:
            continue
        lines.append(
            f"| {DISPLAY[m]} | {ms([r['T_nll']['T'] for r in rs], 1, 2)} "
            f"| {_fmt_pair(np.mean([r['raw']['ece'] for r in rs]), np.mean([r['scaled_nll']['ece'] for r in rs]), 100)} "
            f"| {_fmt_pair(np.mean([r['raw']['adaptive_ece'] for r in rs]), np.mean([r['scaled_nll']['adaptive_ece'] for r in rs]), 100)} "
            f"| {_fmt_pair(np.mean([r['raw']['brier'] for r in rs]), np.mean([r['scaled_nll']['brier'] for r in rs]), 1, 4)} "
            f"| {_fmt_pair(np.mean([r['raw']['brier_top1'] for r in rs]), np.mean([r['scaled_nll']['brier_top1'] for r in rs]), 1, 4)} "
            f"| {_fmt_pair(np.mean([r['raw']['nll'] for r in rs]), np.mean([r['scaled_nll']['nll'] for r in rs]), 1, 3)} |")
    crf = load(root, "crf")
    if crf:
        r, s = crf["raw"], crf["scaled_probs"]
        lines.append(
            f"| CRF (sensitivity) | {crf['T_probs']['T']:.2f} "
            f"| {_fmt_pair(r['ece'], s['ece'], 100)} "
            f"| {_fmt_pair(r['adaptive_ece'], s['adaptive_ece'], 100)} "
            f"| {_fmt_pair(r['brier'], s['brier'], 1, 4)} "
            f"| {_fmt_pair(r['brier_top1'], s['brier_top1'], 1, 4)} "
            f"| {_fmt_pair(r['nll'], s['nll'], 1, 3)} |")
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
        "Coverage = fraction of predictions retained at or above the threshold; "
        "risk = error rate among them. Neural rows: mean ± std over 3 seeds.",
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
        for key, label in (("raw", "marginals"), ("scaled_probs", "scaled (T on dev marginals)")):
            ops = {op["tau"]: op for op in crf[key]["operating_points"]}
            lines.append(
                f"| CRF | {label} | {ops[0.70]['coverage']*100:.2f} "
                f"| {ops[0.70]['risk']*100:.2f} | {ops[0.85]['coverage']*100:.2f} "
                f"| {ops[0.85]['risk']*100:.2f} |")
    return "\n".join(lines) + "\n"


def table_routing(root: Path) -> str:
    lines = [
        "# Routing shift under temperature scaling (NLL protocol)",
        "",
        "Labels never change; only the threshold decision does. Neural values "
        "are mean ± SD over the three seeds; Δrisk CIs are sentence-cluster "
        "bootstrap 95% intervals (shown for each seed).",
        "",
        "| Model | τ | Δcov (pp) | cross↓ n | err among↓ (%) | err among kept (%) "
        "| risk raw→scaled (%) | Δrisk@τ, seed CIs (pp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in NEURAL:
        rs = runs(m, root)
        for tau in (0.70, 0.85):
            entries = [r["thresholds_nll"]["routing_shift"][f"tau={tau}"] for r in rs]
            down_n = [e["n_cross_down"] for e in entries]
            d_err = [e["down_error_rate"] for e in entries if e["down_error_rate"] is not None]
            k_err = [e["error_rate_among_always_above"] for e in entries
                     if e["error_rate_among_always_above"] is not None]
            cov_d = [e["coverage_delta_pp"] for e in entries]
            dr = [r["delta_risk_sentence"][f"tau={tau}"] for r in rs]
            ci_txt = "; ".join(
                f"seed {r['run'].split('_')[1]}: [{d['ci95_low']*100:+.2f}, {d['ci95_high']*100:+.2f}]"
                for r, d in zip(rs, dr))
            lines.append(
                f"| {DISPLAY[m]} | {tau} | {np.mean(cov_d):+.2f} "
                f"| {np.mean(down_n):.1f} ± {np.std(down_n, ddof=1):.1f} "
                f"| {np.mean(d_err)*100:.1f} ± {np.std(d_err, ddof=1)*100:.1f} "
                f"| {np.mean(k_err)*100:.1f} ± {np.std(k_err, ddof=1)*100:.1f} "
                f"| {np.mean([e['risk_raw'] for e in entries])*100:.2f} → "
                f"{np.mean([e['risk_scaled'] for e in entries])*100:.2f} "
                f"| {ci_txt} |")
    crf = load(root, "crf")
    if crf:
        for tau in (0.70, 0.85):
            e = crf["thresholds_raw"]["routing_shift"][f"tau={tau}"]
            lines.append(
                f"| CRF (T on marginals) | {tau} | {e['coverage_delta_pp']:+.2f} "
                f"| {e['n_cross_down']} | {e['down_error_rate']*100:.1f} "
                f"| {e['error_rate_among_always_above']*100:.1f} "
                f"| {e['risk_raw']*100:.2f} → {e['risk_scaled']*100:.2f} | — |")
    return "\n".join(lines) + "\n"


def table_per_tag(root: Path) -> str:
    lines = [
        "# Per-tag calibration (descriptive; neural = seed means, CRF single model)",
        "",
        "| UPOS | model | n | acc (%) | mean conf (%) | gap (pp) |",
        "|---|---|---|---|---|---|",
    ]
    sources = []
    for m in NEURAL:
        rs = runs(m, root)
        if not rs:
            continue
        tags = {}
        for t in {t for r in rs for t in r["raw"]["per_tag"]}:
            entries = [r["raw"]["per_tag"][t] for r in rs if t in r["raw"]["per_tag"]]
            tags[t] = {
                "n": entries[0]["n"],
                "acc": float(np.mean([e["acc"] for e in entries])),
                "mean_conf": float(np.mean([e["mean_conf"] for e in entries])),
                "gap_pp": float(np.mean([e["gap_pp"] for e in entries])),
            }
        sources.append((DISPLAY[m], tags))
    crf = load(root, "crf")
    if crf:
        sources.append(("CRF", crf["raw"]["per_tag"]))

    def tag_n(sources, t) -> int:
        for _, pt in sources:
            if t in pt and pt[t]["n"]:
                return pt[t]["n"]
        return 0

    tags = sorted({t for _, pt in sources for t in pt}, key=lambda t: -tag_n(sources, t))
    for t in tags:
        for name, pt in sources:
            e = pt.get(t)
            if not e or e["n"] == 0:
                continue
            lines.append(f"| {t} | {name} | {e['n']} | {e['acc']*100:.1f} "
                         f"| {e['mean_conf']*100:.1f} | {e['gap_pp']:+.1f} |")
    return "\n".join(lines) + "\n"


def _ci(b: dict, key: str, scale: float, prec: int) -> str:
    e = b.get(key)
    if not e:
        return "—"
    return (f"{e['ci95_low']*scale:.{prec}f}–{e['ci95_high']*scale:.{prec}f} "
            f"({e['mean']*scale:.{prec}f})")


def _bootstrap_table(root: Path, sentence: bool) -> str:
    unit = "sentence-cluster" if sentence else "token-level"
    bk_raw = "bootstrap_sentence_raw" if sentence else "bootstrap_raw"
    bk_scl = "bootstrap_sentence_scaled_nll" if sentence else "bootstrap_scaled_nll"
    lines = [
        f"# Bootstrap 95% CIs ({unit}, 1,000 resamples, test split, seed-42 runs)",
        "",
        f"| Model | conf | acc (%) | macro F1 | ECE (%) | Brier top-1 | NLL "
        "| cov@0.85 (%) | risk@0.85 (%) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in NEURAL:
        r = load(root, f"{m}_42")
        if not r:
            continue
        for key, label in ((bk_raw, "raw"), (bk_scl, "scaled (NLL)")):
            b = r[key]
            lines.append(
                f"| {DISPLAY[m]} | {label} | {_ci(b, 'accuracy', 100, 1)} "
                f"| {_ci(b, 'macro_f1', 100, 1)} | {_ci(b, 'ece', 100, 2)} "
                f"| {_ci(b, 'brier_top1', 1, 4)} | {_ci(b, 'nll', 1, 3)} "
                f"| {_ci(b, 'coverage@0.85', 100, 2)} | {_ci(b, 'risk@0.85', 100, 2)} |")
    crf = load(root, "crf")
    if crf:
        b = crf[bk_raw]
        lines.append(f"| CRF | marginals | {_ci(b, 'accuracy', 100, 1)} "
                     f"| {_ci(b, 'macro_f1', 100, 1)} | {_ci(b, 'ece', 100, 2)} "
                     f"| {_ci(b, 'brier_top1', 1, 4)} | {_ci(b, 'nll', 1, 3)} "
                     f"| {_ci(b, 'coverage@0.85', 100, 2)} | {_ci(b, 'risk@0.85', 100, 2)} |")
        if "bootstrap_sentence_scaled_probs" in crf:
            b = crf["bootstrap_sentence_scaled_probs"]
            lines.append(f"| CRF | scaled marginals | {_ci(b, 'accuracy', 100, 1)} "
                         f"| {_ci(b, 'macro_f1', 100, 1)} | {_ci(b, 'ece', 100, 2)} "
                         f"| {_ci(b, 'brier_top1', 1, 4)} | {_ci(b, 'nll', 1, 3)} "
                         f"| {_ci(b, 'coverage@0.85', 100, 2)} | {_ci(b, 'risk@0.85', 100, 2)} |")
    note = ("Sentences are resampled with replacement; every draw keeps all "
            "tokens of each sampled sentence, so within-sentence dependence is "
            "preserved (primary protocol)."
            if sentence else
            "Tokens are resampled independently — does not preserve "
            "within-sentence dependence; retained as a sensitivity analysis.")
    lines += ["", note, "",
              "Full CI sets (all metrics, all seeds) live in results/metrics/*.json "
              "under bootstrap_sentence_* / bootstrap_*."]
    return "\n".join(lines) + "\n"


def table_ranking(root: Path) -> str:
    """Does scaling change the confidence ordering? (R2-5) + cross-fitted T."""
    lines = [
        "# Ranking invariance under temperature scaling + cross-fitted T",
        "",
        "Spearman ρ between raw and scaled confidence orderings; AURC before "
        "and after scaling; cross-fitted T (5 sentence folds within dev) and "
        "the resulting test ECE range. Neural rows: seed means (ranges in the "
        "metrics JSONs).",
        "",
        "| Model | AURC raw | AURC scaled | Spearman ρ | T (full dev) "
        "| T (cross-fitted range) | test ECE under fold Ts (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in NEURAL:
        rs = runs(m, root)
        if not rs:
            continue
        ece_lo = min(r["crossfit_T"]["test_ece_range"][0] for r in rs) * 100
        ece_hi = max(r["crossfit_T"]["test_ece_range"][1] for r in rs) * 100
        lines.append(
            f"| {DISPLAY[m]} | {ms([r['ranking']['aurc_raw'] for r in rs], 1, 4)} "
            f"| {ms([r['ranking']['aurc_scaled'] for r in rs], 1, 4)} "
            f"| {ms([r['ranking']['spearman_raw_vs_scaled'] for r in rs], 1, 4)} "
            f"| {ms([r['T_nll']['T'] for r in rs], 1, 2)} "
            f"| {np.mean([r['crossfit_T']['T_range'][0] for r in rs]):.2f}–"
            f"{np.mean([r['crossfit_T']['T_range'][1] for r in rs]):.2f} "
            f"| {ece_lo:.2f}–{ece_hi:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    root = Path(__file__).resolve().parent.parent / "results"
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tables = (("summary", table_summary),
              ("post_ts", table_post_ts),
              ("operating_points", table_operating),
              ("routing", table_routing),
              ("per_tag", table_per_tag),
              ("bootstrap_sentence", lambda r: _bootstrap_table(r, True)),
              ("bootstrap_token", lambda r: _bootstrap_table(r, False)),
              ("ranking", table_ranking))
    for name, fn in tables:
        path = out / f"{name}.md"
        path.write_text(fn(root), encoding="utf-8")
        print(f"wrote tables/{name}.md")


if __name__ == "__main__":
    main()
