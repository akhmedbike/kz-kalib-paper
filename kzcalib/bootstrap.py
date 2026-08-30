"""Token-level bootstrap confidence intervals.

Replaces the fold-level CIs that the small corpus cannot afford: resample
test tokens (the paper's replacement strategy, named in Limitations).
"""

from __future__ import annotations

import numpy as np

from .metrics import (adaptive_ece, aurc, brier, brier_top1, conf_of,
                      coverage_at, ece, nll, selective_risk_at)


def bootstrap_metrics(probs: np.ndarray, gold_ids: np.ndarray,
                      taus=(0.70, 0.85), n_resamples: int = 1000,
                      seed: int = 42, n_bins: int = 10) -> dict:
    """Percentile 95% CIs for the headline scalar metrics."""
    rng = np.random.default_rng(seed)
    n = len(gold_ids)
    conf = conf_of(probs)
    correct = probs.argmax(axis=1) == gold_ids

    keys = ["accuracy", "ece", "adaptive_ece", "brier", "brier_top1", "nll",
            "aurc", "mean_conf"] + [f"coverage@{t}" for t in taus] + \
           [f"risk@{t}" for t in taus]
    samples: dict[str, list] = {k: [] for k in keys}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        p, g, c, k = probs[idx], gold_ids[idx], conf[idx], correct[idx]
        samples["accuracy"].append(float(k.mean()))
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
    out["_meta"] = {"n_resamples": n_resamples, "seed": seed, "n_tokens": n}
    return out
