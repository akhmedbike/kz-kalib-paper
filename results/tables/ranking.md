# Ranking invariance under temperature scaling + cross-fitted T

Spearman ρ between raw and scaled confidence orderings; AURC before and after scaling; cross-fitted T (5 sentence folds within dev) and the resulting test ECE range. Neural rows: seed means (ranges in the metrics JSONs).

| Model | AURC raw | AURC scaled | Spearman ρ | T (full dev) | T (cross-fitted range) | test ECE under fold Ts (%) |
|---|---|---|---|---|---|---|
| KazBERT | 0.0187 ± 0.0012 | 0.0191 ± 0.0013 | 0.9994 ± 0.0003 | 1.43 ± 0.07 | 1.41–1.45 | 2.36–3.07 |
| KazRoBERTa | 0.0213 ± 0.0015 | 0.0218 ± 0.0015 | 0.9977 ± 0.0008 | 1.64 ± 0.17 | 1.61–1.67 | 1.46–2.60 |
| XLM-R | 0.0152 ± 0.0010 | 0.0153 ± 0.0011 | 0.9985 ± 0.0005 | 1.46 ± 0.06 | 1.45–1.47 | 1.70–2.91 |
