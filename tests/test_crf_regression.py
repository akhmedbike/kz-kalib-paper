"""Regression test: the released CRF through the kzcalib harness.

Pins the harness to the measured values on the canonical test split
(2026-08-27, this machine). Cross-check against the paper draft:

    paper    harness  metric
    94.48    94.54    accuracy          (matches within platform drift)
    2.71     2.65     ECE %             (matches within platform drift)
    0.044    0.0440   brier_top1        (exact: the paper number was the
                                         binary top-1 Brier, NOT the Eq.-4
                                         multiclass Brier, which is 0.0915)
    92.97/   92.97/   coverage/risk @0.85 (exact)
    96.74/   96.74/   coverage/risk @0.70 (exact)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path  # noqa: E402

from kzcalib.crf import run_crf  # noqa: E402
from kzcalib.data import load_split  # noqa: E402
from kzcalib.metrics import (brier, brier_top1, conf_of,  # noqa: E402
                             selective_risk_at)


def test_crf_matches_paper():
    data = load_split(Path(__file__).resolve().parent.parent / "data")
    out = run_crf(data["test"])

    probs, gold_ids = out["probs"], out["gold_ids"]
    conf = conf_of(probs)
    correct = probs.argmax(axis=1) == gold_ids

    acc = float(correct.mean())
    b_top1 = brier_top1(conf, correct)
    b_full = brier(probs, gold_ids)
    op85 = selective_risk_at(conf, correct, 0.85)
    op70 = selective_risk_at(conf, correct, 0.70)

    assert len(gold_ids) == 1594, f"expected 1594 test tokens, got {len(gold_ids)}"
    assert abs(acc - 0.9454) < 0.002, f"accuracy drifted: {acc:.4f}"
    assert abs(b_top1 - 0.0440) < 0.002, f"brier_top1 drifted: {b_top1:.4f}"
    assert abs(b_full - 0.0915) < 0.005, f"brier_full drifted: {b_full:.4f}"
    assert abs(op85["coverage"] - 0.9297) < 0.002
    assert abs(op85["risk"] - 0.0344) < 0.002
    assert abs(op70["coverage"] - 0.9674) < 0.002
    assert abs(op70["risk"] - 0.0415) < 0.002
