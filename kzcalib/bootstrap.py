"""Bootstrap confidence intervals.

Two resampling units, both over the test split:

- ``bootstrap_metrics``: token-level resampling (the original protocol; keeps
  the Limitations discussion honest but understates uncertainty because
  tokens within a sentence are not independent).
- ``bootstrap_metrics_sentence``: sentence-level cluster bootstrap — resample
  the 162 test sentences with replacement and keep every token of each drawn
  sentence. This is the primary protocol for the revision; the token-level
  variant is retained as a sensitivity analysis.
"""

from __future__ import annotations

import numpy as np

from .metrics import (adaptive_ece, aurc, brier, brier_top1, conf_of,
                      coverage_at, ece, macro_f1_ids, nll, selective_risk_at)


def _ci_block(samples: dict[str, list]) -> dict:
    out = {}
    for k, vals in samples.items():
        v = np.array(vals, dtype=np.float64)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out[k] = None
            continue
        out[k] = {
            "mean": float(v.mean()),
            "ci95_low": float(np.percentile(v, 2.5)),
            "ci95_high": float(np.percentile(v, 97.5)),
        }
    return out


def _resample_metrics(probs: np.ndarray, gold_ids: np.ndarray, idx: np.ndarray,
                      taus, n_bins: int, samples: dict[str, list]) -> None:
    """Fill ``samples`` with every headline metric computed on ``idx``."""
    p, g = probs[idx], gold_ids[idx]
    c = p.max(axis=1)
    pred = p.argmax(axis=1)
    k = pred == g
    samples["accuracy"].append(float(k.mean()))
    samples["macro_f1"].append(macro_f1_ids(pred, g, p.shape[1]))
    samples["ece"].append(ece(c, k, n_bins))
    samples["adaptive_ece"].append(adaptive_ece(c, k, n_bins))
    samples["brier"].append(brier(p, g))
    samples["brier_top1"].append(brier_top1(c, k))
    samples["nll"].append(nll(p, g))
    samples["aurc"].append(aurc(c, k))
    samples["mean_conf"].append(float(c.mean()))
    for t in taus:
        samples[f"coverage@{t}"].append(coverage_at(c, t))
        r = selective_risk_at(c, k, t)["risk"]
        samples[f"risk@{t}"].append(r if r is not None else np.nan)


def _keys(taus) -> list[str]:
    return (["accuracy", "macro_f1", "ece", "adaptive_ece", "brier", "brier_top1",
             "nll", "aurc", "mean_conf"]
            + [f"coverage@{t}" for t in taus] + [f"risk@{t}" for t in taus])


def bootstrap_metrics(probs: np.ndarray, gold_ids: np.ndarray,
                      taus=(0.70, 0.85), n_resamples: int = 1000,
                      seed: int = 42, n_bins: int = 10) -> dict:
    """Percentile 95% CIs, token-level resampling (sensitivity analysis)."""
    rng = np.random.default_rng(seed)
    n = len(gold_ids)
    samples: dict[str, list] = {k: [] for k in _keys(taus)}
    for _ in range(n_resamples):
        _resample_metrics(probs, gold_ids,
                          rng.integers(0, n, size=n), taus, n_bins, samples)
    out = _ci_block(samples)
    out["_meta"] = {"unit": "token", "n_resamples": n_resamples, "seed": seed,
                    "n_tokens": n}
    return out


def sentence_index(sent_ids: list[str]) -> np.ndarray:
    """Map per-token sentence ids to a compact integer cluster index."""
    uniq = {s: i for i, s in enumerate(dict.fromkeys(sent_ids))}
    return np.array([uniq[s] for s in sent_ids], dtype=np.int64)


def bootstrap_metrics_sentence(probs: np.ndarray, gold_ids: np.ndarray,
                               sent_ids: list[str], taus=(0.70, 0.85),
                               n_resamples: int = 1000, seed: int = 42,
                               n_bins: int = 10) -> dict:
    """Percentile 95% CIs, sentence-level cluster bootstrap (primary).

    Resamples the test sentences with replacement; every draw keeps all tokens
    of each sampled sentence, so within-sentence dependence is preserved.
    """
    rng = np.random.default_rng(seed)
    cluster = sentence_index(sent_ids)
    n_sent = int(cluster.max()) + 1
    tokens_of = [np.flatnonzero(cluster == c) for c in range(n_sent)]
    samples: dict[str, list] = {k: [] for k in _keys(taus)}
    for _ in range(n_resamples):
        pick = rng.integers(0, n_sent, size=n_sent)
        idx = np.concatenate([tokens_of[c] for c in pick])
        _resample_metrics(probs, gold_ids, idx, taus, n_bins, samples)
    out = _ci_block(samples)
    out["_meta"] = {"unit": "sentence", "n_resamples": n_resamples,
                    "seed": seed, "n_sentences": n_sent, "n_tokens": len(gold_ids)}
    return out


def bootstrap_delta_risk_sentence(conf_raw: np.ndarray, conf_scaled: np.ndarray,
                                  correct: np.ndarray, sent_ids: list[str],
                                  tau: float = 0.85, n_resamples: int = 1000,
                                  seed: int = 42) -> dict:
    """Sentence-cluster bootstrap CI for the change in selective risk at tau
    caused by temperature scaling (risk_scaled − risk_raw; negative = safer)."""
    rng = np.random.default_rng(seed)
    cluster = sentence_index(sent_ids)
    n_sent = int(cluster.max()) + 1
    tokens_of = [np.flatnonzero(cluster == c) for c in range(n_sent)]
    deltas = []
    for _ in range(n_resamples):
        pick = rng.integers(0, n_sent, size=n_sent)
        idx = np.concatenate([tokens_of[c] for c in pick])
        r_raw = selective_risk_at(conf_raw[idx], correct[idx], tau)["risk"]
        r_scl = selective_risk_at(conf_scaled[idx], correct[idx], tau)["risk"]
        if r_raw is not None and r_scl is not None:
            deltas.append(r_scl - r_raw)
    v = np.array(deltas)
    return {
        "tau": tau,
        "mean": float(v.mean()),
        "ci95_low": float(np.percentile(v, 2.5)),
        "ci95_high": float(np.percentile(v, 97.5)),
        "_meta": {"unit": "sentence", "n_resamples": n_resamples, "seed": seed},
    }
