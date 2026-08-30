# Routing shift under temperature scaling (NLL protocol)

Labels never change; only the threshold decision does.

| Model | τ | cov raw (%) | cov scaled (%) | Δcov (pp) | cross↓ n | err among↓ (%) | err among kept (%) | risk raw (%) | risk scaled (%) |
|---|---|---|---|---|---|---|---|---|---|
| KazBERT | 0.7 | 97.09 | 94.00 | -3.09 | 49.3 | 43.73 | 6.50 | 7.67 | 6.50 |
| KazBERT | 0.85 | 94.23 | 89.42 | -4.81 | 76.7 | 33.39 | 5.07 | 6.52 | 5.07 |
| KazRoBERTa | 0.7 | 96.65 | 91.99 | -4.66 | 74.3 | 43.14 | 5.82 | 7.59 | 5.82 |
| KazRoBERTa | 0.85 | 93.54 | 85.30 | -8.24 | 131.3 | 31.43 | 3.92 | 6.32 | 3.92 |
| XLM-R | 0.7 | 97.91 | 95.46 | -2.45 | 39.0 | 38.99 | 5.41 | 6.26 | 5.41 |
| XLM-R | 0.85 | 95.75 | 90.86 | -4.89 | 78.0 | 32.39 | 4.03 | 5.46 | 4.03 |
