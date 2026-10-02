"""CRF baseline evaluated through the same harness as the neural taggers.

The released model (``crf_baseline/crf_model.joblib``, c1=0.2/c2=0.01,
trained on the canonical train split) is loaded as-is; features are built by
running the baseline's Stage-1 RMA over each token and calling its
``CRFTagger.extract_features``. Confidence = marginal probability of the
predicted tag (``predict_marginals``). The whole baseline is bundled in
``crf_baseline/`` — no external checkout is required.
"""

from __future__ import annotations

import os

import joblib
import numpy as np

from .crf_baseline.crf_tagger import CRFTagger
from .crf_baseline.rma import ReverseMorphAnalyzer
from .data import LABEL2ID, UPOS_LABELS, Sentence


def crf_model_path() -> str:
    return os.path.join(os.path.dirname(__file__), "crf_baseline", "crf_model.joblib")


def run_crf(sentences: list[Sentence]) -> dict:
    """Score a split with the released CRF.

    Returns per-token records and the full 17-class probability matrix built
    from the marginals (absent tags get 0), so Brier/NLL use the same code
    path as the neural dumps.
    """
    model = joblib.load(crf_model_path())
    rma = ReverseMorphAnalyzer()

    records = []
    probs_rows: list[np.ndarray] = []

    for sent in sentences:
        rma_results = [rma.analyze(tok) for tok in sent.tokens]
        feats = [CRFTagger.extract_features(rma_results, i)
                 for i in range(len(rma_results))]
        preds = model.predict([feats])[0]
        marginals = model.predict_marginals([feats])[0]

        for i, (tok, gold, pred) in enumerate(zip(sent.tokens, sent.upos, preds)):
            row = np.zeros(len(UPOS_LABELS), dtype=np.float64)
            for tag, p in marginals[i].items():
                if tag in LABEL2ID:
                    row[LABEL2ID[tag]] = p
            # Marginals may not sum to exactly 1 across the 17-label space;
            # renormalise defensively.
            s = row.sum()
            if s > 0:
                row /= s
            probs_rows.append(row)
            records.append({
                "sent_id": sent.sent_id,
                "tok_idx": i,
                "token": tok,
                "gold": gold,
                "pred": pred,
                "conf": float(row[LABEL2ID[pred]] if pred in LABEL2ID else 0.0),
            })

    probs = np.vstack(probs_rows)
    return {"records": records, "probs": probs,
            "gold_ids": np.array([LABEL2ID[r["gold"]] for r in records])}
