"""Regression test: the released CRF through the kzcalib harness.

Pins the harness to the measured values on the canonical test split for the
retrained CRF baseline (trained, like every other data source it touches, on
the 754-sentence training partition only — the RMA lemma dictionary is built
from the same partition, so dev/test material is excluded from training and
feature construction alike):

    acc 83.31, ECE 13.66%, Brier top-1 0.1456, cov/risk 92.91/13.91 @0.85.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path  # noqa: E402

from kzcalib.crf import run_crf  # noqa: E402
from kzcalib.data import load_split  # noqa: E402
from kzcalib.metrics import (brier, brier_top1, conf_of,  # noqa: E402
                             ece, selective_risk_at)


def test_crf_matches_released_model():
    data = load_split(Path(__file__).resolve().parent.parent / "data")
    out = run_crf(data["test"])

    probs, gold_ids = out["probs"], out["gold_ids"]
    conf = conf_of(probs)
    correct = probs.argmax(axis=1) == gold_ids

    acc = float(correct.mean())
    b_top1 = brier_top1(conf, correct)
    b_full = brier(probs, gold_ids)
    e = ece(conf, correct, 10)
    op85 = selective_risk_at(conf, correct, 0.85)
    op70 = selective_risk_at(conf, correct, 0.70)

    assert len(gold_ids) == 1594, f"expected 1594 test tokens, got {len(gold_ids)}"
    assert abs(acc - 0.8331) < 0.002, f"accuracy drifted: {acc:.4f}"
    assert abs(b_top1 - 0.1456) < 0.002, f"brier_top1 drifted: {b_top1:.4f}"
    assert abs(b_full - 0.3011) < 0.005, f"brier_full drifted: {b_full:.4f}"
    assert abs(e - 0.1366) < 0.003, f"ece drifted: {e:.4f}"
    assert abs(op85["coverage"] - 0.9291) < 0.002
    assert abs(op85["risk"] - 0.1391) < 0.003
    assert abs(op70["coverage"] - 0.9630) < 0.002
    assert abs(op70["risk"] - 0.1505) < 0.003
