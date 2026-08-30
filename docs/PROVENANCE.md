# Provenance

## Data

- `data/{train,dev,test}.conllu` — the decontaminated canonical split:
  seed 42, 85/15/15 of the 1,078-sentence pool → 754/162/162 sentences,
  7,376/1,566/1,594 syntactic tokens. SHA256 in `data/SHA256SUMS`. This is
  exactly the split in Table 1 of the draft (`data/split_meta.json` holds
  the split metadata).
- Tokenisation for evaluation: syntactic words only (multiword-token ranges
  skipped), matching `kazmorph/scripts/train_crf.py`.

## Models

- **KazBERT** `Eraly-ml/KazBERT`, **KazRoBERTa** `kz-transformers/kaz-roberta-conversational`,
  **XLM-R base** `FacebookAI/xlm-roberta-base` — fine-tuned here on train,
  first-subword labelling, unified hyperparameters (lr 5e-5, batch 16,
  AdamW, grad clip 1.0, ≤40 epochs, early stop patience 5 on dev token
  accuracy), seeds 13/42/123. Best-epoch state saved per run in
  `results/checkpoints/`. Base weights come from the local HF cache; no
  downloads at run time (`HF_HUB_OFFLINE=1`).
- **CRF** — the released baseline model from the authors' prior toolkit,
  bundled in full under `kzcalib/crf_baseline/` (sklearn-crfsuite, lbfgs,
  c1=0.2, c2=0.01, max_iter 150, all_possible_states/transitions; trained
  on the same train split — see `crf_hyperparams.json`). Features built by
  running the toolkit's Stage-1 RMA per token and `CRFTagger.extract_features`.
  Loaded as-is, never retrained here. The RMA lemma dictionary is built at
  runtime from the toolkit's original (pre-decontamination) train file,
  bundled as `crf_baseline/train_split.conllu`.

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

## Numbers cross-checked against the draft (CRF, unchanged)

| quantity | draft | this repo | status |
|---|---|---|---|
| test tokens | 1,594 | 1,594 | exact |
| accuracy | 94.48% | 94.54% | same run, platform drift |
| ECE | 2.71% | 2.65% | same run, platform drift |
| Brier (top-1) | 0.044 | 0.0440 | exact |
| cov/risk @0.85 | 92.97% / 3.44% | 92.97% / 3.44% | exact |
| cov/risk @0.70 | 96.74% / 4.15% | 96.74% / 4.15% | exact |

## Environment

- Apple M3 Pro, 18 GB, macOS 15, MPS; Python 3.14.4; torch 2.13.0,
  transformers 5.16.1, numpy 2.x. `PYTHONHASHSEED=42` for all stages.
- Deterministic algorithms enabled with `warn_only=True` (MPS).
