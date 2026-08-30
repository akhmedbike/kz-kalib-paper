# Licenses for the bundled CRF baseline

This directory bundles the released CRF baseline so the repository is
self-contained. The files originate from the authors' prior morphological
toolkit (anonymised for review) and from third-party sources; licenses apply
per file:

## MIT (project-original, the authors' prior toolkit)

- `affixes.py` — Kazakh affix definitions
- `rma.py` — Stage-1 reverse morphological analyzer
- `crf_tagger.py` — Stage-2 CRF tagger / feature extraction
- `crf_model.joblib` — the released CRF model (trained with
  `crf_hyperparams.json`: sklearn-crfsuite, lbfgs, c1=0.2, c2=0.01,
  max_iter 150, all_possible_states/transitions, seed 42)
- `crf_hyperparams.json` — training hyperparameters
- `lexicons_extra.py` — manual lexicon corrections

## GNU GPL v2 or later (derived from Apertium-kaz)

- `lexicons.py` — lexicon sets auto-generated from the Apertium-kaz lexc
  dictionary (https://github.com/apertium/apertium-kaz), bundled unchanged.
  Files that embed this material are distributed under GPLv2+ terms.

## CC BY-SA 4.0 (derived from UD Kazakh-KTB)

- `train_split.conllu` — the toolkit's original (pre-decontamination) UD
  Kazakh-KTB train file, used to build the RMA lemma dictionary at runtime.
  See `data/README.md` in the repository root for the treebank attribution
  and required citations.
