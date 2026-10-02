# Proper scoring rules, raw → temperature-scaled (test split)

T fitted on development data by NLL (transformers: logits; CRF: marginals). Neural rows: seed means. Accuracy is unchanged by scaling.

| Model | T | ECE (%) | aECE (%) | Brier full | Brier top-1 | NLL |
|---|---|---|---|---|---|---|
| KazBERT | 1.43 ± 0.07 | 6.43 → 2.74 | 6.34 → 3.29 | 0.1555 → 0.1463 | 0.0739 → 0.0669 | 0.444 → 0.369 |
| KazRoBERTa | 1.64 ± 0.17 | 6.51 → 2.14 | 6.26 → 2.29 | 0.1558 → 0.1440 | 0.0736 → 0.0645 | 0.470 → 0.359 |
| XLM-R | 1.46 ± 0.06 | 5.45 → 2.14 | 5.17 → 2.14 | 0.1277 → 0.1203 | 0.0609 → 0.0553 | 0.380 → 0.310 |
| CRF (sensitivity) | 2.22 | 13.66 → 4.02 | 13.66 → 3.94 | 0.3011 → 0.2741 | 0.1456 → 0.1258 | 1.092 → 0.669 |
