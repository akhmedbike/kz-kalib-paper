"""Temperature fitting, two protocols.

- ``fit_nll``  (primary, submission-quality): minimise dev NLL, evaluate on test.
- ``fit_ece_grid`` (legacy): grid 0.8–5.0 step 0.1 minimising dev ECE — the
  protocol of the original stored artifacts, kept for the sensitivity analysis.
"""

from __future__ import annotations

import numpy as np

from .metrics import ece, nll, softmax


def fit_nll(logits: np.ndarray, gold_ids: np.ndarray,
            lo: float = 0.05, hi: float = 10.0) -> dict:
    """Golden-section search for T minimising NLL of the gold class."""
    logits = np.asarray(logits, dtype=np.float64)
    gr = (np.sqrt(5.0) + 1.0) / 2.0
    a, b = lo, hi
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    for _ in range(200):
        if abs(b - a) < 1e-6:
            break
        fc = nll(softmax(logits, c), gold_ids)
        fd = nll(softmax(logits, d), gold_ids)
        if fc < fd:
            b, d = d, c
            c = b - (b - a) / gr
        else:
            a, c = c, d
            d = a + (b - a) / gr
    T = (a + b) / 2
    return {"T": float(T), "criterion": "dev_nll"}


def fit_ece_grid(logits: np.ndarray, gold_ids: np.ndarray,
                 grid=None) -> dict:
    """Legacy protocol: grid search minimising dev ECE (10 equal-width bins)."""
    if grid is None:
        grid = np.round(np.arange(0.8, 5.0 + 0.001, 0.1), 2)
    logits = np.asarray(logits, dtype=np.float64)
    conf = softmax(logits, 1.0).max(axis=1)
    correct = logits.argmax(axis=1) == gold_ids  # argmax is T-invariant
    best_T, best_ece = None, np.inf
    for T in grid:
        v = ece(softmax(logits, T).max(axis=1), correct)
        if v < best_ece:
            best_T, best_ece = float(T), v
    return {"T": best_T, "criterion": "dev_ece_grid", "dev_ece_at_T": float(best_ece)}
