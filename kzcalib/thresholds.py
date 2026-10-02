"""Threshold-local calibration and routing-shift analysis (RQ3).

Temperature scaling preserves every predicted label, so accuracy is invariant —
but the *decision* at a fixed threshold is not. These functions quantify what
happens exactly where the deployed corrector reads the confidence.
"""

from __future__ import annotations

import numpy as np

from .metrics import ece, selective_risk_at


def band_calibration(conf: np.ndarray, correct: np.ndarray,
                     lo: float, hi: float) -> dict:
    """Calibration inside a confidence band [lo, hi] — where the thresholds live."""
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=np.float64)
    m = (conf >= lo) & (conf < hi)
    n = int(m.sum())
    if n == 0:
        return {"band": [lo, hi], "n": 0}
    return {
        "band": [lo, hi],
        "n": n,
        "share": n / len(conf),
        "mean_conf": float(conf[m].mean()),
        "accuracy": float(correct[m].mean()),
        "gap_pp": float((conf[m].mean() - correct[m].mean()) * 100),
        "local_ece": ece(conf[m], correct[m], n_bins=5),
    }


def routing_shift(conf_raw: np.ndarray, conf_scaled: np.ndarray,
                  taus=(0.70, 0.85)) -> dict:
    """How many tokens cross each threshold after scaling, labels unchanged.

    A token that stays above tau keeps its automatic decision; a token that
    drops below tau moves to review. ``down`` tokens change the system's
    behaviour despite identical predictions.
    """
    conf_raw = np.asarray(conf_raw)
    conf_scaled = np.asarray(conf_scaled)
    out = {}
    for tau in taus:
        above_raw = conf_raw >= tau
        above_scaled = conf_scaled >= tau
        down = above_raw & ~above_scaled     # loses automatic handling -> review
        up = ~above_raw & above_scaled       # gains automatic handling
        out[f"tau={tau}"] = {
            "coverage_raw": float(above_raw.mean()),
            "coverage_scaled": float(above_scaled.mean()),
            "coverage_delta_pp": float((above_scaled.mean() - above_raw.mean()) * 100),
            "n_cross_down": int(down.sum()),
            "n_cross_up": int(up.sum()),
        }
    return out


def routing_shift_full(conf_raw: np.ndarray, conf_scaled: np.ndarray,
                       correct: np.ndarray, taus=(0.70, 0.85)) -> dict:
    """routing_shift plus the error composition of the crossing tokens."""
    conf_raw = np.asarray(conf_raw)
    conf_scaled = np.asarray(conf_scaled)
    correct = np.asarray(correct, dtype=bool)
    base = routing_shift(conf_raw, conf_scaled, taus)
    for tau in taus:
        entry = base[f"tau={tau}"]
        above_raw = conf_raw >= tau
        above_scaled = conf_scaled >= tau
        down = above_raw & ~above_scaled
        up = ~above_raw & above_scaled
        entry["down_error_rate"] = float((~correct[down]).mean()) if down.any() else None
        entry["up_error_rate"] = float((~correct[up]).mean()) if up.any() else None
        entry["risk_raw"] = selective_risk_at(conf_raw, correct, tau)["risk"]
        entry["risk_scaled"] = selective_risk_at(conf_scaled, correct, tau)["risk"]
        kept = above_raw & above_scaled
        entry["error_rate_among_always_above"] = (
            float((~correct[kept]).mean()) if kept.any() else None
        )
    return base
