"""Calibration and selective-prediction metrics.

Every function is pure NumPy over per-token arrays so that the same code
serves the CRF marginals and the neural softmax vectors, and so that
bootstrap resampling (kzcalib.bootstrap) can index straight into them.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Row-wise softmax of logits divided by temperature T."""
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def conf_of(probs: np.ndarray) -> np.ndarray:
    """Max softmax probability (the confidence the decision layer consumes)."""
    return probs.max(axis=1)


def pred_of(probs: np.ndarray, id2label: dict) -> list[str]:
    return [id2label[i] for i in probs.argmax(axis=1)]


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width expected calibration error (Eq. 3 of the paper, M = 10)."""
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = in_bin.sum()
        if cnt == 0:
            continue
        total += (cnt / n) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total)


def adaptive_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE over equal-mass bins (robust to the thin low-confidence tail)."""
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(conf, kind="stable")
    splits = np.array_split(order, n_bins)
    n = len(conf)
    return float(sum((len(s) / n) * abs(correct[s].mean() - conf[s].mean())
                     for s in splits if len(s)))


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Per-bin (count, mean confidence, accuracy) for reliability diagrams."""
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(in_bin.sum())
        bins.append({
            "bin": i,
            "center": (lo + hi) / 2,
            "count": cnt,
            "mean_conf": float(conf[in_bin].mean()) if cnt else None,
            "accuracy": float(correct[in_bin].mean()) if cnt else None,
        })
    return bins


def brier(probs: np.ndarray, gold_ids: np.ndarray) -> float:
    """Full-vector Brier score (Eq. 4): mean over tokens of sum_k (p_ik - y_ik)^2."""
    probs = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(gold_ids)), gold_ids] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def brier_top1(conf: np.ndarray, correct: np.ndarray) -> float:
    """Binary Brier on the top-1 confidence: mean (conf - correct)^2.

    This is what the draft's reported 0.044 actually was (verified to 4
    decimals against the released CRF); kept alongside the Eq.-4 multiclass
    Brier so the paper can name both explicitly.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    return float(((conf - correct) ** 2).mean())


def nll(probs: np.ndarray, gold_ids: np.ndarray, eps: float = 1e-12) -> float:
    """Negative log-likelihood of the gold class (Eq. 4)."""
    p = np.asarray(probs, dtype=np.float64)[np.arange(len(gold_ids)), gold_ids]
    return float(-np.log(np.clip(p, eps, None)).mean())


def per_tag_calibration(gold: list[str], pred: list[str], conf: np.ndarray):
    """Descriptive per-tag calibration (Table 4 style)."""
    conf = np.asarray(conf)
    out = {}
    for tag in sorted(set(gold)):
        m = np.array([g == tag for g in gold])
        n = int(m.sum())
        acc = float((np.array(pred)[m] == np.asarray(gold)[m]).mean()) if n else None
        mc = float(conf[m].mean()) if n else None
        out[tag] = {
            "n": n,
            "acc": acc,
            "mean_conf": mc,
            "gap_pp": (mc - acc) * 100 if n else None,  # + = over-confident
        }
    return out


# ---------------------------------------------------------------------------
# Selective prediction (Eq. 6)
# ---------------------------------------------------------------------------


def coverage_at(conf: np.ndarray, tau: float) -> float:
    return float((np.asarray(conf) >= tau).mean())


def selective_risk_at(conf: np.ndarray, correct: np.ndarray, tau: float):
    """(coverage, risk, n_auto) at threshold tau; risk = error among auto-decided."""
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=bool)
    keep = conf >= tau
    n = int(keep.sum())
    if n == 0:
        return {"tau": tau, "coverage": 0.0, "risk": None, "n_auto": 0}
    return {
        "tau": tau,
        "coverage": float(keep.mean()),
        "risk": float((~correct[keep]).mean()),
        "n_auto": n,
    }


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray, n_points: int = 101):
    """Risk among the most-confident q fraction of tokens, q in (0, 1]."""
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - correct[order]
    cum_err = np.cumsum(err)
    counts = np.arange(1, len(conf) + 1)
    coverages = counts / len(conf)
    idx = np.unique(np.round(np.linspace(1, len(conf), n_points)).astype(int) - 1)
    return {
        "coverage": coverages[idx].tolist(),
        "risk": (cum_err[idx] / counts[idx]).tolist(),
    }


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    curve = risk_coverage_curve(conf, correct, n_points=len(conf))
    c = np.array(curve["coverage"])
    r = np.array(curve["risk"])
    return float(np.trapezoid(r, c))


def full_report(probs: np.ndarray, gold_ids: np.ndarray, id2label: dict,
                taus=(0.70, 0.85), n_bins: int = 10) -> dict:
    """Every metric the paper reports, from one probability matrix."""
    conf = conf_of(probs)
    pred_ids = probs.argmax(axis=1)
    correct = pred_ids == gold_ids
    pred = [id2label[i] for i in pred_ids]
    gold = [id2label[i] for i in gold_ids]
    rep = {
        "n": int(len(gold_ids)),
        "accuracy": float(correct.mean()),
        "macro_f1": _macro_f1(gold, pred),
        "ece": ece(conf, correct, n_bins),
        "adaptive_ece": adaptive_ece(conf, correct, n_bins),
        "brier": brier(probs, gold_ids),
        "brier_top1": brier_top1(conf, correct),
        "nll": nll(probs, gold_ids),
        "aurc": aurc(conf, correct),
        "mean_conf": float(conf.mean()),
        "reliability_bins": reliability_bins(conf, correct, n_bins),
        "per_tag": per_tag_calibration(gold, pred, conf),
        "operating_points": [selective_risk_at(conf, correct, t) for t in taus],
        "risk_coverage": risk_coverage_curve(conf, correct),
    }
    return rep


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    tags = sorted(set(gold))
    f1s = []
    for t in tags:
        tp = sum(1 for g, p in zip(gold, pred) if g == t and p == t)
        fp = sum(1 for g, p in zip(gold, pred) if g != t and p == t)
        fn = sum(1 for g, p in zip(gold, pred) if g == t and p != t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_f1_ids(pred_ids: np.ndarray, gold_ids: np.ndarray,
                 n_labels: int) -> float:
    """Vectorised macro F1 over the labels observed in ``gold_ids``.

    Same convention as ``_macro_f1``: averaged over the UPOS labels present in
    the gold split (SYM, absent from the test gold, is excluded; a label never
    predicted contributes F1 = 0). Fast enough for bootstrap resampling.
    """
    pred_ids = np.asarray(pred_ids)
    gold_ids = np.asarray(gold_ids)
    present = np.zeros(n_labels, dtype=bool)
    present[gold_ids] = True
    conf = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(conf, (gold_ids, pred_ids), 1)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(f1[present].mean())
