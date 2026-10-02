# Licenses for the bundled CRF baseline

This directory bundles the CRF baseline so the repository is self-contained.
The code originates from the authors' prior morphological toolkit
(anonymised for review); the model is retrained for this study on the
canonical split (`scripts/train_crf.py`). Third-party sources carry their own
licenses, per file:

## MIT (project-original, the authors' prior toolkit)

- `affixes.py` — Kazakh affix definitions
- `rma.py` — Stage-1 reverse morphological analyzer
- `crf_tagger.py` — Stage-2 CRF tagger / feature extraction
- `crf_model.joblib` — the CRF model retrained on the canonical training
  partition (trained with `crf_hyperparams.json`: sklearn-crfsuite, lbfgs,
  c1=0.05, c2=0.05 selected on dev, max_iter 150,
  all_possible_states/transitions, seed 42)
- `crf_hyperparams.json` — training hyperparameters and selection protocol
- `lexicons_extra.py` — manual lexicon corrections

## GNU GPL v2 or later (derived from Apertium-kaz)

- `lexicons.py` — lexicon sets auto-generated from the Apertium-kaz lexc
  dictionary (https://github.com/apertium/apertium-kaz), bundled unchanged.
  Files that embed this material are distributed under GPLv2+ terms.

## CC BY-SA 4.0 (derived from UD Kazakh-KTB)

- `train_split.conllu` — a byte-identical copy of the canonical
  `data/train.conllu` (754 training sentences), used to build the RMA lemma
  dictionary at runtime so that the baseline also runs standalone. It
  contains no development or test material. See `data/README.md` in the
  repository root for the treebank attribution and required citations.
