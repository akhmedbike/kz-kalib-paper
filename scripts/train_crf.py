#!/usr/bin/env python3
"""Train the CRF baseline for the calibration study.

Protocol (identical to the correction pipeline's original trainer, re-run on
the canonical split so that every data source the CRF touches — weights,
feature lexicons, and the RMA lemma dictionary — is built from the 754
training sentences only):

  - Stage-1 features come from the bundled Reverse Morphological Analyzer,
    whose UD lemma dictionary is built from ``data/train.conllu`` alone;
  - the CRF (sklearn-crfsuite, lbfgs, max_iter 150, all_possible_states and
    transitions) is trained on ``data/train.conllu``;
  - hyperparameters are selected from a fixed 8-point grid by development-set
    weighted F1 — never on the test split;
  - seed 42.

Writes ``kzcalib/crf_baseline/crf_model.joblib`` and ``crf_hyperparams.json``.

Usage:
    python scripts/train_crf.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics

from kzcalib.crf_baseline.crf_tagger import CRFTagger
from kzcalib.crf_baseline.rma import ReverseMorphAnalyzer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAM_GRID = [
    {"c1": 0.01, "c2": 0.01},
    {"c1": 0.01, "c2": 0.1},
    {"c1": 0.05, "c2": 0.01},
    {"c1": 0.05, "c2": 0.05},
    {"c1": 0.1, "c2": 0.01},
    {"c1": 0.1, "c2": 0.1},
    {"c1": 0.2, "c2": 0.01},
    {"c1": 0.5, "c2": 0.01},
]


def parse_conllu(filepath: str) -> list[list[dict]]:
    """Parse CoNLL-U into sentences of token dicts (same rules as the pipeline)."""
    sentences, current = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if "-" in parts[0] or "." in parts[0]:  # multiword tokens / empty nodes
                continue
            if len(parts) >= 4:
                current.append({"form": parts[1], "upos": parts[3]})
    if current:
        sentences.append(current)
    return sentences


def prepare_data(sentences: list[list[dict]], rma: ReverseMorphAnalyzer):
    X, y = [], []
    for sent in sentences:
        rma_results = [rma.analyze(tok["form"]) for tok in sent]
        feats = [CRFTagger.extract_features(rma_results, i)
                 for i in range(len(rma_results))]
        X.append(feats)
        y.append([tok["upos"] for tok in sent])
    return X, y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-file", default=os.path.join(PROJECT_ROOT, "data", "train.conllu"))
    p.add_argument("--dev-file", default=os.path.join(PROJECT_ROOT, "data", "dev.conllu"))
    args = p.parse_args()

    random.seed(args.seed)

    train_sents = parse_conllu(args.train_file)
    dev_sents = parse_conllu(args.dev_file)
    print(f"train: {len(train_sents)} sentences, "
          f"dev: {len(dev_sents)} sentences")

    rma = ReverseMorphAnalyzer()
    print(f"RMA lemma dictionary: {len(rma.lemma_dict)} forms "
          f"(built from the canonical train split only)")

    X_train, y_train = prepare_data(train_sents, rma)
    X_dev, y_dev = prepare_data(dev_sents, rma)
    print(f"train tokens: {sum(len(s) for s in X_train)}, "
          f"dev tokens: {sum(len(s) for s in X_dev)}")

    best_f1, best_params, best_model = 0.0, {}, None
    for params in PARAM_GRID:
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=params["c1"],
            c2=params["c2"],
            max_iterations=150,
            all_possible_transitions=True,
            all_possible_states=True,
        )
        crf.fit(X_train, y_train)
        f1 = metrics.flat_f1_score(y_dev, crf.predict(X_dev), average="weighted")
        acc = metrics.flat_accuracy_score(y_dev, crf.predict(X_dev))
        marker = "  *best*" if f1 > best_f1 else ""
        print(f"  c1={params['c1']:<5} c2={params['c2']:<5} "
              f"acc={acc:.4f}  w-F1={f1:.4f}{marker}")
        if f1 > best_f1:
            best_f1, best_params, best_model = f1, params, crf

    out_dir = os.path.join(PROJECT_ROOT, "kzcalib", "crf_baseline")
    model_path = os.path.join(out_dir, "crf_model.joblib")
    joblib.dump(best_model, model_path)

    with open(os.path.join(out_dir, "crf_hyperparams.json"), "w", encoding="utf-8") as f:
        json.dump({
            "algorithm": "lbfgs",
            "c1": best_params["c1"],
            "c2": best_params["c2"],
            "max_iterations": 150,
            "all_possible_transitions": True,
            "all_possible_states": True,
            "dev_weighted_f1": best_f1,
            "grid": PARAM_GRID,
            "train_file": "data/train.conllu",
            "dev_file": "data/dev.conllu",
            "lemma_dict_source": "data/train.conllu (canonical train split only)",
            "seed": args.seed,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nbest: c1={best_params['c1']}, c2={best_params['c2']}, "
          f"dev w-F1={best_f1:.4f}")
    print(f"model written to {model_path}")


if __name__ == "__main__":
    main()
