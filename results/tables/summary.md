# Table 3 (harmonised): tagging and calibration, one split, one code path

Test split N = 1,594 tokens; temperature fitted on dev (N = 1,566): primary protocol NLL, legacy protocol ECE grid (0.8–5.0, step 0.1). Neural rows: mean ± std over 3 seeds. acc = token accuracy.

| Model | acc (%) | macro F1 | T(NLL) | ECE raw (%) | ECE scaled NLL (%) | ECE scaled grid (%) | Brier full | Brier top-1 | NLL | AURC |
|---|---|---|---|---|---|---|---|---|---|---|
| KazBERT | 91.05 ± 0.14 | 80.21 ± 1.62 | 1.43 ± 0.07 | 6.43 ± 0.35 | 2.74 ± 0.09 | 2.67 ± 0.38 | 0.1555 ± 0.0045 | 0.0739 ± 0.0028 | 0.4444 ± 0.0377 | 0.0187 ± 0.0012 |
| KazRoBERTa | 90.90 ± 0.38 | 82.16 ± 1.49 | 1.64 ± 0.17 | 6.51 ± 0.88 | 2.14 ± 0.19 | 1.60 ± 0.07 | 0.1558 ± 0.0080 | 0.0736 ± 0.0043 | 0.4702 ± 0.0663 | 0.0213 ± 0.0015 |
| XLM-R | 92.83 ± 0.43 | 84.31 ± 2.38 | 1.46 ± 0.06 | 5.45 ± 0.62 | 2.14 ± 0.40 | 2.01 ± 0.30 | 0.1277 ± 0.0093 | 0.0609 ± 0.0049 | 0.3802 ± 0.0220 | 0.0152 ± 0.0010 |
| CRF | 94.54 | 89.14 | — | 2.65 | — | — | 0.0915 | 0.0440 | 0.2280 | 0.0088 |

Accuracy is identical raw vs scaled (arg-max preserving).
