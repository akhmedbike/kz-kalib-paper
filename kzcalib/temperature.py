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


def softmax_T_probs(probs: np.ndarray, T: float) -> np.ndarray:
    """Temperature scaling applied to probabilities instead of logits.

    p_T ∝ p^(1/T): equivalent to logit-space scaling when p came from a
    softmax, and the natural analogue for models that expose marginals only
    (the CRF). Like logit scaling, it never changes the arg-max.
    """
    p = np.asarray(probs, dtype=np.float64) ** (1.0 / T)
    return p / p.sum(axis=1, keepdims=True)


def fit_nll_probs(probs: np.ndarray, gold_ids: np.ndarray,
                  lo: float = 0.05, hi: float = 10.0) -> dict:
    """Fit T on probability vectors (marginals) by minimising NLL."""
    probs = np.asarray(probs, dtype=np.float64)
    grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 4)
    best_T, best = None, np.inf
    for T in grid:
        v = nll(softmax_T_probs(probs, float(T)), gold_ids)
        if v < best:
            best_T, best = float(T), v
    return {"T": best_T, "criterion": "dev_nll_probs", "dev_nll_at_T": best}


def cross_fitted_T(logits: np.ndarray, gold_ids: np.ndarray,
                   sent_ids: list[str], n_folds: int = 5,
                   seed: int = 42) -> dict:
    """Cross-fitted temperature within the development set (sensitivity).

    Splits the dev sentences into ``n_folds`` groups; for each fold, fits T by
    NLL on the remaining folds. Reports the per-fold T estimates and the test
    metrics are evaluated by the caller under each fold's T. Quantifies how
    much the dual use of dev (early stopping + T fitting) can move the
    temperature without retraining.
    """
    from .bootstrap import sentence_index

    cluster = sentence_index(sent_ids)
    n_sent = int(cluster.max()) + 1
    rng = np.random.default_rng(seed)
    sent_fold = rng.permutation(n_sent) % n_folds
    fold_of_token = sent_fold[cluster]
    out = {"n_folds": n_folds, "seed": seed, "folds": []}
    Ts = []
    for f in range(n_folds):
        tr = fold_of_token != f
        fitted = fit_nll(logits[tr], gold_ids[tr])
        Ts.append(fitted["T"])
        out["folds"].append({"fold": f, "T": fitted["T"],
                             "n_dev_tokens_fit": int(tr.sum())})
    out["T_values"] = Ts
    out["T_mean"] = float(np.mean(Ts))
    out["T_range"] = [float(min(Ts)), float(max(Ts))]
    return out
