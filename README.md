# kz-calib: Trustworthy Confidence for Kazakh Morphological Tagging

Code and measured results for the paper *"Trustworthy Confidence for Kazakh
Morphological Tagging: Calibration and Selective Decision-Making in a
Low-Resource Text-Correction Pipeline"*.

One code path measures four taggers — KazBERT, KazRoBERTa, XLM-R base and a
feature-based CRF — on the canonical 754/162/162 split of
UD_Kazakh-KTB (7,376 / 1,566 / 1,594 tokens, seed 42, decontaminated), with
temperature fitted on dev and every calibration/selective-prediction metric
evaluated on the held-out test split.

## Layout

```
configs/        per-model hyperparameters (unified: lr 5e-5, batch 16, ≤40 ep,
                early stop on dev accuracy, patience 5)
data/           canonical UD_Kazakh-KTB split + SHA256SUMS
                (CC BY-SA 4.0, see data/README.md)
kzcalib/
  data.py       CoNLL-U parsing (syntactic words only), 17-UPOS label space
  train.py      manual fine-tuning loop (MPS), first-subword pooling,
                per-token logit dumps for dev/test
  crf.py        the CRF baseline through the same harness;
                confidence = predict_marginals of the pred tag
  crf_baseline/ bundled CRF baseline (no external dependencies):
                Stage-1 RMA, lexicons, the retrained model + hyperparams
  metrics.py    ECE / adaptive ECE / Brier (full & top-1) / NLL / AURC /
                reliability bins / per-tag calibration / coverage / selective
                risk / risk-coverage curves
  temperature.py fit T on dev: NLL (primary), legacy ECE grid, probability-
                space T for CRF marginals, cross-fitted T within dev
  bootstrap.py  sentence-cluster bootstrap 95% CIs (primary) + token-level
                variant (sensitivity), Δrisk CIs (1,000 resamples)
  thresholds.py threshold-local band calibration + routing-shift analysis
  analyze.py    dumps -> results/metrics/{run}.json
  aggregate.py  metrics -> results/tables/*.md
  figures.py    metrics -> results/figures/* (PNG 300 dpi + vector PDF/SVG)
scripts/        run_all.sh (full benchmark), train_crf.py (CRF retraining),
                make_fig1.py (Figure 1 pipeline diagram, regenerated as code)
tests/          CRF regression test pinning the paper's CRF row
```

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/run_all.sh          # 3 models × 3 seeds, CRF, analysis (caffeinate-wrapped)
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m kzcalib.aggregate && .venv/bin/python -m kzcalib.figures
```

Requirements: macOS with MPS (or CPU fallback) and the three HF base models in
the local HF cache (`Eraly-ml/KazBERT`, `kz-transformers/kaz-roberta-conversational`,
`FacebookAI/xlm-roberta-base`). The CRF baseline is bundled in
`kzcalib/crf_baseline/` and needs no external checkout.

Individual stages:

```bash
.venv/bin/python -m kzcalib.train --config configs/xlmr.yaml --seed 42
.venv/bin/python -m kzcalib.analyze --run xlmr_42
.venv/bin/python -m kzcalib.analyze --crf
.venv/bin/python scripts/train_crf.py     # retrain the CRF baseline (minutes)
```

## Provenance notes

- The neural checkpoints behind the original draft's Table 3 no longer exist;
  this repository retrains all three transformers on the exact Table 1 split
  and replaces the stored-artifact numbers with a harmonised benchmark
  (see docs/PROVENANCE.md).
- **The CRF is retrained on the canonical split** (`scripts/train_crf.py`,
  seed 42, hyperparameters selected on dev). The originally released
  checkpoint's RMA lemma dictionary had been built from a pre-decontamination
  file overlapping most of the test split; every data source the bundled CRF
  touches is now built from the 754 training sentences only. The draft's CRF
  row (94.54% accuracy, 2.65% ECE) was an artifact of that leakage; the clean
  values are 83.31% and 13.66%.
- The draft's reported CRF "Brier 0.044" was the **binary top-1** Brier of the
  leaked-dictionary model; the revision reports both Brier variants of the
  retrained model, named explicitly.
- Long-running stages are wrapped in `caffeinate -is` so the machine does not
  sleep mid-experiment.

## Archive

This repository holds the code, the canonical data split, the CRF baseline,
the per-run metric JSONs (`results/metrics/`), and the aggregated tables
(`results/tables/`). The large generated artifacts are archived on Zenodo
([doi:10.5281/zenodo.23096850](https://doi.org/10.5281/zenodo.23096850)):

- per-token probability dumps for every reported run (dev + test, 3 models ×
  3 seeds) — `kz-calib-v2.zip`;
- the nine fine-tuned transformer checkpoints — split archives
  `kz-calib-v2-ckpt-{kazbert,kazroberta}.zip` and
  `kz-calib-v2-ckpt-xlmr-{13,42,123}.zip` (SHA-256 checksums in
  `MANIFEST-checkpoints.sha256`, reload instructions in `CHECKPOINTS.md`);
- rendered figures (PNG 300 dpi + vector PDF/SVG).

## License

- Code: MIT — see [LICENSE](LICENSE).
- Data (`data/`): CC BY-SA 4.0 — derived from the UD_Kazakh-KTB treebank;
  see [data/README.md](data/README.md) for attribution and citations.
- CRF baseline (`kzcalib/crf_baseline/`): mixed — MIT code and model,
  GPLv2+ lexicons derived from [Apertium-kaz](https://github.com/apertium/apertium-kaz),
  CC BY-SA 4.0 `train_split.conllu`; see
  [kzcalib/crf_baseline/LICENSES.md](kzcalib/crf_baseline/LICENSES.md).

Not committed (regenerated by the commands above): model checkpoints
(`results/checkpoints/`), per-token logit dumps (`results/dumps/`) and
rendered figures (`results/figures/`). Committed results: per-run metrics
(`results/metrics/`) and the aggregated tables (`results/tables/`).
