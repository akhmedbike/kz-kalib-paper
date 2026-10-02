# Operating points on the deployed thresholds

Coverage = fraction of predictions retained at or above the threshold; risk = error rate among them. Neural rows: mean ± std over 3 seeds.

| Model | conf source | cov@0.70 (%) | risk@0.70 (%) | cov@0.85 (%) | risk@0.85 (%) |
|---|---|---|---|---|---|
| KazBERT | raw | 97.09 ± 0.61 | 7.67 ± 0.23 | 94.23 ± 0.60 | 6.52 ± 0.64 |
| KazBERT | scaled (T by dev NLL) | 94.00 ± 0.43 | 6.50 ± 0.36 | 89.42 ± 0.38 | 5.07 ± 0.49 |
| KazRoBERTa | raw | 96.65 ± 0.84 | 7.59 ± 0.43 | 93.54 ± 1.67 | 6.32 ± 0.65 |
| KazRoBERTa | scaled (T by dev NLL) | 91.99 ± 1.23 | 5.82 ± 0.32 | 85.30 ± 1.92 | 3.92 ± 0.64 |
| XLM-R | raw | 97.91 ± 0.51 | 6.26 ± 0.78 | 95.75 ± 0.31 | 5.46 ± 0.62 |
| XLM-R | scaled (T by dev NLL) | 95.46 ± 0.35 | 5.41 ± 0.55 | 90.86 ± 0.51 | 4.03 ± 0.21 |
| CRF | marginals | 96.30 | 15.05 | 92.91 | 13.91 |
| CRF | scaled (T on dev marginals) | 85.70 | 12.81 | 67.63 | 9.74 |
