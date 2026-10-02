# Table 3 (harmonised): tagging and raw calibration, one split, one code path

Test split N = 1,594 tokens. Neural rows: mean ± std over 3 seeds. acc = token accuracy; aECE = equal-mass adaptive ECE.

| Model | acc (%) | macro F1 | ECE raw (%) | aECE raw (%) | Brier full | Brier top-1 | NLL | AURC |
|---|---|---|---|---|---|---|---|---|
| KazBERT | 91.05 ± 0.14 | 80.21 ± 1.62 | 6.43 ± 0.35 | 6.34 ± 0.25 | 0.1555 ± 0.0045 | 0.0739 ± 0.0028 | 0.4444 ± 0.0377 | 0.0187 ± 0.0012 |
| KazRoBERTa | 90.90 ± 0.38 | 82.16 ± 1.49 | 6.51 ± 0.88 | 6.26 ± 1.13 | 0.1558 ± 0.0080 | 0.0736 ± 0.0043 | 0.4702 ± 0.0663 | 0.0213 ± 0.0015 |
| XLM-R | 92.83 ± 0.43 | 84.31 ± 2.38 | 5.45 ± 0.62 | 5.17 ± 0.56 | 0.1277 ± 0.0093 | 0.0609 ± 0.0049 | 0.3802 ± 0.0220 | 0.0152 ± 0.0010 |
| CRF | 83.31 | 75.85 | 13.66 | 13.66 | 0.3011 | 0.1456 | 1.0920 | 0.0829 |

Accuracy is identical raw vs scaled (arg-max preserving).
