# Provenance

## Data

- `data/{train,dev,test}.conllu` — the decontaminated canonical split:
  seed 42, 85/15/15 of the 1,078-sentence pool → 754/162/162 sentences,
  7,376/1,566/1,594 syntactic tokens. SHA256 in `data/SHA256SUMS`. This is
  exactly the split in Table 1 of the draft (`data/split_meta.json` holds
  the split metadata).
- Tokenisation for evaluation: syntactic words only (multiword-token ranges
  skipped), matching the CRF baseline's original training parser
  (`scripts/train_crf.py` in this repository).

## Models

- **KazBERT** `Eraly-ml/KazBERT`, **KazRoBERTa** `kz-transformers/kaz-roberta-conversational`,
  **XLM-R base** `FacebookAI/xlm-roberta-base` — fine-tuned here on train,
  first-subword labelling, unified hyperparameters (lr 5e-5, batch 16,
  AdamW, grad clip 1.0, ≤40 epochs, early stop patience 5 on dev token
  accuracy), seeds 13/42/123. Best-epoch state saved per run in
  `results/checkpoints/`. Base weights come from the local HF cache; no
  downloads at run time (`HF_HUB_OFFLINE=1`).
- **CRF** — the correction pipeline's baseline architecture, **retrained for
  this study on the canonical split** with `scripts/train_crf.py`
  (sklearn-crfsuite, lbfgs, max_iter 150, all_possible_states/transitions;
  hyperparameters c1=0.05, c2=0.05 selected on the development split by
  weighted F1 over a fixed 8-point grid; seed 42 — see
  `kzcalib/crf_baseline/crf_hyperparams.json`). Features built by running the
  Stage-1 RMA per token and `CRFTagger.extract_features`. **Every data source
  the CRF touches is built from the 754 training sentences only**: the RMA
  lemma dictionary is loaded from `data/train.conllu` (a byte-identical copy
  is bundled as `crf_baseline/train_split.conllu` for standalone use), and
  the treebank-independent lexicons come from the Apertium-kaz-derived
  `lexicons.py`. Development and test material is excluded from training,
  feature construction, and hyperparameter selection alike.

## What was replaced relative to the draft (and why)

1. **Neural Table 3 numbers.** The original F1/ECE/T values came from stored
   artifacts whose checkpoints and per-token outputs no longer exist on any
   accessible machine (verified 2026-08-27 across all available prior
   checkouts). They are replaced by a harmonised re-run:
   same split, same measurement code as the CRF, three seeds, CIs.
2. **Temperature protocol.** Primary: T fitted on dev by minimising NLL,
   evaluated on test (the draft's own "submission-quality" protocol). The
   legacy grid protocol (ECE on dev, 0.8–5.0 step 0.1) is reported alongside
   as a protocol-sensitivity analysis.
3. **"Brier 0.044".** Verified to be the binary top-1 Brier
   mean(conf − correct)², exact to 4 decimals. The Eq.-4 multiclass Brier of
   the same CRF run is 0.0915. The revision reports both, named explicitly.
4. **CRF retrained (revision).** The draft evaluated the previously released
   CRF checkpoint, whose RMA lemma dictionary had been built from a
   pre-decontamination training file that overlapped 138 of the 162 test
   sentences — i.e. the dictionary, and therefore the Stage-1 features, had
   seen most of the test material. For the revision the dictionary is rebuilt
   from the canonical training partition alone and the CRF retrained on the
   same partition with the original protocol. The draft's CRF row
   (94.54% accuracy, 2.65% ECE) was an artifact of that leakage; the clean
   values are 83.31% and 13.66%, with a sensitivity experiment adding a
   dev-fitted temperature on the marginals (ECE 4.02%). All CRF-dependent
   tables, figures, and regression-test pins were regenerated.

## CRF numbers, draft vs this repo

| quantity | draft (leaked dict) | this repo (clean) |
|---|---|---|
| test tokens | 1,594 | 1,594 |
| accuracy | 94.54% | 83.31% |
| ECE | 2.65% | 13.66% |
| Brier (top-1) | 0.0440 | 0.1456 |
| Brier (multiclass) | 0.0915 | 0.3011 |
| NLL | 0.228 | 1.092 |
| cov/risk @0.85 | 92.97% / 3.44% | 92.91% / 13.91% |
| cov/risk @0.70 | 96.74% / 4.15% | 96.30% / 15.05% |
| macro F1 | 89.1 | 75.9 |

## Environment

- Apple M3 Pro, 18 GB, macOS 15, MPS; Python 3.14.4; torch 2.13.0,
  transformers 5.16.1, numpy 2.x. `PYTHONHASHSEED=42` for all stages.
- Deterministic algorithms enabled with `warn_only=True` (MPS).
