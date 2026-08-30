"""Turn per-token dumps into the paper's measured results.

For each neural run (model × seed):
  - fit T on dev logits by NLL (primary) and by the legacy ECE grid,
  - evaluate raw and scaled on the held-out test split,
  - bootstrap CIs, threshold-local bands, routing-shift under both T protocols.

The CRF is analysed raw-only (marginals, not logits — no temperature), exactly
as in the paper.

Usage:
    python -m kzcalib.analyze --run kazbert_42        # neural dumps
    python -m kzcalib.analyze --crf                   # CRF baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bootstrap import bootstrap_metrics
from .data import ID2LABEL
from .metrics import conf_of, full_report
from .temperature import fit_ece_grid, fit_nll
from .thresholds import band_calibration, routing_shift_full

BANDS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.85),
         (0.85, 0.90), (0.90, 1.001)]
TAUS = (0.70, 0.85)


def load_dump(path: str | Path) -> dict:
    path = Path(path)
    logits, gold_ids, tokens, golds = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            logits.append(r["logits"])
            gold_ids.append(r["gold_id"])
            tokens.append(r["token"])
            golds.append(r["gold"])
    return {
        "logits": np.array(logits, dtype=np.float64),
        "gold_ids": np.array(gold_ids, dtype=np.int64),
        "tokens": tokens,
        "golds": golds,
    }


def analyze_split(probs: np.ndarray, gold_ids: np.ndarray) -> dict:
    return full_report(probs, gold_ids, ID2LABEL, taus=TAUS)


def threshold_block(logits: np.ndarray, gold_ids: np.ndarray, T: float) -> dict:
    from .metrics import softmax
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


def analyze_neural(run: str, results_root: str | Path = "results") -> dict:
    root = Path(results_root)
    dev = load_dump(root / "dumps" / f"{run}_dev.jsonl")
    test = load_dump(root / "dumps" / f"{run}_test.jsonl")

    out: dict = {"run": run, "n_dev": len(dev["gold_ids"]), "n_test": len(test["gold_ids"])}

    # -- raw test report -----------------------------------------------------
    from .metrics import softmax
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

    # -- bootstrap CIs (raw and primary NLL-scaled) ---------------------------
    out["bootstrap_raw"] = bootstrap_metrics(probs_raw, test["gold_ids"], TAUS)
    out["bootstrap_scaled_nll"] = bootstrap_metrics(
        softmax(test["logits"], out["T_nll"]["T"]), test["gold_ids"], TAUS)

    path = root / "metrics" / f"{run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {path}")
    return out


def analyze_crf(crf_out: dict, results_root: str | Path = "results") -> dict:
    """CRF: raw marginals only, same measurement code path."""
    probs, gold_ids = crf_out["probs"], crf_out["gold_ids"]
    out = {"run": "crf", "n_test": int(len(gold_ids))}
    out["raw"] = analyze_split(probs, gold_ids)
    out["bootstrap_raw"] = bootstrap_metrics(probs, gold_ids, TAUS)
    conf = conf_of(probs)
    correct = probs.argmax(axis=1) == gold_ids
    out["thresholds_raw"] = {
        "bands_raw": [band_calibration(conf, correct, lo, hi) for lo, hi in BANDS],
    }
    root = Path(results_root)
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
        from .crf import run_crf
        from .data import load_split
        data = load_split(Path(__file__).resolve().parent.parent / "data")
        crf_out = run_crf(data["test"])
        analyze_crf(crf_out, args.results_root)
    if not args.run and not args.crf:
        p.error("need --run or --crf")


if __name__ == "__main__":
    main()
