# Routing shift under temperature scaling (NLL protocol)

Labels never change; only the threshold decision does. Neural values are mean ± SD over the three seeds; Δrisk CIs are sentence-cluster bootstrap 95% intervals (shown for each seed).

| Model | τ | Δcov (pp) | cross↓ n | err among↓ (%) | err among kept (%) | risk raw→scaled (%) | Δrisk@τ, seed CIs (pp) |
|---|---|---|---|---|---|---|---|
| KazBERT | 0.7 | -3.09 | 49.3 ± 5.0 | 43.7 ± 7.7 | 6.5 ± 0.4 | 7.67 → 6.50 | seed 13: [-1.56, -0.50]; seed 42: [-1.86, -0.75]; seed 123: [-1.93, -0.64] |
| KazBERT | 0.85 | -4.81 | 76.7 ± 4.2 | 33.4 ± 2.5 | 5.1 ± 0.5 | 6.52 → 5.07 | seed 13: [-2.36, -0.90]; seed 42: [-2.18, -0.96]; seed 123: [-1.92, -0.69] |
| KazRoBERTa | 0.7 | -4.66 | 74.3 ± 7.6 | 43.1 ± 8.3 | 5.8 ± 0.3 | 7.59 → 5.82 | seed 13: [-2.34, -1.18]; seed 42: [-2.69, -1.31]; seed 123: [-2.18, -0.94] |
| KazRoBERTa | 0.85 | -8.24 | 131.3 ± 6.8 | 31.4 ± 6.2 | 3.9 ± 0.6 | 6.32 → 3.92 | seed 13: [-3.20, -1.73]; seed 42: [-3.74, -1.99]; seed 123: [-2.70, -1.29] |
| XLM-R | 0.7 | -2.45 | 39.0 ± 2.6 | 39.0 ± 7.5 | 5.4 ± 0.6 | 6.26 → 5.41 | seed 13: [-1.20, -0.28]; seed 42: [-1.71, -0.61]; seed 123: [-1.18, -0.31] |
| XLM-R | 0.85 | -4.89 | 78.0 ± 7.0 | 32.4 ± 9.4 | 4.0 ± 0.2 | 5.46 → 4.03 | seed 13: [-2.01, -0.84]; seed 42: [-2.57, -1.17]; seed 123: [-1.74, -0.45] |
| CRF (T on marginals) | 0.7 | -10.60 | 169 | 33.1 | 12.8 | 15.05 → 12.81 | — |
| CRF (T on marginals) | 0.85 | -25.28 | 403 | 25.1 | 9.7 | 13.91 → 9.74 | — |
