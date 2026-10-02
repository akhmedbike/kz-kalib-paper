"""Turn per-token dumps into the paper's measured results.

For each neural run (model × seed):
  - fit T on dev logits by NLL (primary) and by the legacy ECE grid,
  - evaluate raw and scaled on the held-out test split,
  - sentence-level cluster bootstrap CIs (primary) and token-level bootstrap
    (sensitivity), threshold-local bands, routing-shift under both T protocols,
  - cross-fitted T within dev and its effect on the test metrics,
  - ranking diagnostics: does scaling change the confidence ordering (AURC
    raw vs scaled, Spearman rho between raw and scaled confidence).

The CRF is analysed raw (marginals as deployed) plus a probability-space
temperature fitted on dev marginals as a sensitivity experiment.

Usage:
    python -m kzcalib.analyze --run kazbert_42        # neural dumps
    python -m kzcalib.analyze --crf                   # CRF baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bootstrap import (bootstrap_delta_risk_sentence,
                        bootstrap_metrics, bootstrap_metrics_sentence)
from .data import ID2LABEL
from .metrics import aurc, conf_of, full_report, softmax
from .temperature import (cross_fitted_T, fit_ece_grid, fit_nll,
                          fit_nll_probs, softmax_T_probs)
from .thresholds import band_calibration, routing_shift_full

BANDS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.85),
         (0.85, 0.90), (0.90, 1.001)]
TAUS = (0.70, 0.85)


def load_dump(path: str | Path) -> dict:
    path = Path(path)
    logits, gold_ids, tokens, golds, sent_ids = [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            logits.append(r["logits"])
            gold_ids.append(r["gold_id"])
            tokens.append(r["token"])
            golds.append(r["gold"])
            sent_ids.append(r["sent_id"])
    return {
        "logits": np.array(logits, dtype=np.float64),
        "gold_ids": np.array(gold_ids, dtype=np.int64),
        "tokens": tokens,
        "golds": golds,
        "sent_ids": sent_ids,
    }


def analyze_split(probs: np.ndarray, gold_ids: np.ndarray) -> dict:
    return full_report(probs, gold_ids, ID2LABEL, taus=TAUS)


def threshold_block(logits: np.ndarray, gold_ids: np.ndarray, T: float) -> dict:
    probs_raw = softmax(logits, 1.0)
    probs_scl = softmax(logits, T)
    conf_raw, conf_scl = conf_of(probs_raw), conf_of(probs_scl)
    correct = probs_raw.argmax(axis=1) == gold_ids  # T-invariant
    return {
        "T": T,
        "bands_raw": [band_calibration(conf_raw, correct, lo, hi)
                      for lo, hi in BANDS],
        "bands_scaled": [band_calibration(conf_scl, correct, lo, hi)
                         for lo, hi in BANDS],
        "routing_shift": routing_shift_full(conf_raw, conf_scl, correct, TAUS),
    }


def ranking_block(logits: np.ndarray, gold_ids: np.ndarray, T: float) -> dict:
    """Does the scalar temperature change the confidence *ordering*?

    For K > 2 the max-softmax is not a monotone function of a shared rescaling
    of the logits, so raw and scaled confidence can in principle reorder two
    tokens. In practice the effect is negligible; these numbers quantify it.
    """
    conf_raw = conf_of(softmax(logits, 1.0))
    conf_scl = conf_of(softmax(logits, T))
    correct = logits.argmax(axis=1) == gold_ids
    # Spearman over distinct-value ranks (average ranks for ties)
    from scipy.stats import spearmanr
    rho = spearmanr(conf_raw, conf_scl).statistic
    return {
        "T": T,
        "aurc_raw": aurc(conf_raw, correct),
        "aurc_scaled": aurc(conf_scl, correct),
        "aurc_delta": aurc(conf_scl, correct) - aurc(conf_raw, correct),
        "spearman_raw_vs_scaled": float(rho),
        "n_conf_ties_raw": int((np.diff(np.sort(conf_raw)) == 0).sum()),
    }


def crossfit_block(dev: dict, test: dict, T_full: float) -> dict:
    """Cross-fitted T within dev + test metrics under each fold's T."""
    cf = cross_fitted_T(dev["logits"], dev["gold_ids"], dev["sent_ids"])
    per_fold_test = []
    for f in cf["folds"]:
        rep = analyze_split(softmax(test["logits"], f["T"]), test["gold_ids"])
        per_fold_test.append({
            "fold": f["fold"], "T": f["T"],
            "test_ece": rep["ece"], "test_nll": rep["nll"],
            "test_brier": rep["brier"], "test_brier_top1": rep["brier_top1"],
        })
    eces = [e["test_ece"] for e in per_fold_test]
    nlls = [e["test_nll"] for e in per_fold_test]
    return {
        **cf,
        "T_full_dev": T_full,
        "per_fold_test": per_fold_test,
        "test_ece_range": [float(min(eces)), float(max(eces))],
        "test_nll_range": [float(min(nlls)), float(max(nlls))],
    }


def analyze_neural(run: str, results_root: str | Path = "results") -> dict:
    root = Path(results_root)
    dev = load_dump(root / "dumps" / f"{run}_dev.jsonl")
    test = load_dump(root / "dumps" / f"{run}_test.jsonl")

    out: dict = {"run": run, "n_dev": len(dev["gold_ids"]), "n_test": len(test["gold_ids"])}

    # -- raw test report -----------------------------------------------------
    probs_raw = softmax(test["logits"], 1.0)
    out["raw"] = analyze_split(probs_raw, test["gold_ids"])

    # -- temperatures fitted on dev, evaluated on test ------------------------
    for key, fitted in (("nll", fit_nll(dev["logits"], dev["gold_ids"])),
                        ("ece_grid", fit_ece_grid(dev["logits"], dev["gold_ids"]))):
        T = fitted["T"]
        probs_T = softmax(test["logits"], T)
        out[f"T_{key}"] = fitted
        out[f"scaled_{key}"] = analyze_split(probs_T, test["gold_ids"])
        out[f"thresholds_{key}"] = threshold_block(test["logits"], test["gold_ids"], T)

    T = out["T_nll"]["T"]

    # -- bootstrap CIs: sentence-cluster (primary) and token-level (sensitivity)
    probs_scl = softmax(test["logits"], T)
    out["bootstrap_sentence_raw"] = bootstrap_metrics_sentence(
        probs_raw, test["gold_ids"], test["sent_ids"], TAUS)
    out["bootstrap_sentence_scaled_nll"] = bootstrap_metrics_sentence(
        probs_scl, test["gold_ids"], test["sent_ids"], TAUS)
    out["bootstrap_raw"] = bootstrap_metrics(probs_raw, test["gold_ids"], TAUS)
    out["bootstrap_scaled_nll"] = bootstrap_metrics(
        probs_scl, test["gold_ids"], TAUS)

    # -- Δrisk under scaling with sentence-cluster CI (per threshold) ---------
    conf_raw, conf_scl = conf_of(probs_raw), conf_of(probs_scl)
    correct = probs_raw.argmax(axis=1) == test["gold_ids"]
    out["delta_risk_sentence"] = {
        f"tau={t}": bootstrap_delta_risk_sentence(
            conf_raw, conf_scl, correct, test["sent_ids"], t)
        for t in TAUS
    }

    # -- cross-fitted T and ranking diagnostics (revision analyses) ----------
    out["crossfit_T"] = crossfit_block(dev, test, T)
    out["ranking"] = ranking_block(test["logits"], test["gold_ids"], T)

    path = root / "metrics" / f"{run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {path}")
    return out


def analyze_crf(results_root: str | Path = "results") -> dict:
    """CRF: raw marginals (deployed form) + dev-fitted probability-space T."""
    from .crf import run_crf
    from .data import load_split

    root = Path(results_root)
    data = load_split(Path(__file__).resolve().parent.parent / "data")
    test_out = run_crf(data["test"])
    dev_out = run_crf(data["dev"])

    probs, gold_ids = test_out["probs"], test_out["gold_ids"]
    out: dict = {"run": "crf", "n_test": int(len(gold_ids)),
                 "n_dev": int(len(dev_out["gold_ids"]))}
    out["raw"] = analyze_split(probs, gold_ids)

    # sentence ids align with load_split order; rebuild them from the data
    test_sids = [s for sent in data["test"] for s in [sent.sent_id] * len(sent.tokens)]
    dev_sids = [s for sent in data["dev"] for s in [sent.sent_id] * len(sent.tokens)]
    out["bootstrap_sentence_raw"] = bootstrap_metrics_sentence(
        probs, gold_ids, test_sids, TAUS)
    out["bootstrap_raw"] = bootstrap_metrics(probs, gold_ids, TAUS)

    # sensitivity (reviewer request): temperature fitted on dev marginals
    fitted = fit_nll_probs(dev_out["probs"], dev_out["gold_ids"])
    out["T_probs"] = fitted
    out["scaled_probs"] = analyze_split(
        softmax_T_probs(probs, fitted["T"]), gold_ids)
    out["bootstrap_sentence_scaled_probs"] = bootstrap_metrics_sentence(
        softmax_T_probs(probs, fitted["T"]), gold_ids, test_sids, TAUS)

    conf = conf_of(probs)
    correct = probs.argmax(axis=1) == gold_ids
    out["thresholds_raw"] = {
        "bands_raw": [band_calibration(conf, correct, lo, hi) for lo, hi in BANDS],
        "routing_shift": routing_shift_full(
            conf, conf_of(softmax_T_probs(probs, fitted["T"])), correct, TAUS),
    }
    out["dev_raw"] = analyze_split(dev_out["probs"], dev_out["gold_ids"])
    out["dev_raw"].pop("risk_coverage", None)

    path = root / "metrics" / "crf.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="neural run id, e.g. kazbert_42")
    p.add_argument("--crf", action="store_true", help="analyse the CRF baseline")
    p.add_argument("--results-root", default="results")
    args = p.parse_args()

    if args.run:
        analyze_neural(args.run, args.results_root)
    if args.crf:
        analyze_crf(args.results_root)
    if not args.run and not args.crf:
        p.error("need --run or --crf")


if __name__ == "__main__":
    main()
