# Bootstrap 95% CI — together-llama-3.3-70b (--prompt cot)

- Sample: 588 (gold, pred) pairs from the §2.15 concordance run
- Bootstrap iterations: 1000, seed: 0
- Gate threshold: 0.85

## Per-class point estimates

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supports | 0.9564 | 0.9599 | 0.9582 | 549 |
| Contradicts | 0.6667 | 0.1538 | 0.2500 | 39 |
| Neutral | 0.0000 | 0.0000 | 0.0000 | 0 |
| Not relevant | 0.0000 | 0.0000 | 0.0000 | 0 |

## Macro-weighted F1 with bootstrap 95% CI

- **Point estimate**: 0.9112 (PASS)
- **95% CI**: [0.8861, 0.9355]  (width 0.0494)
- **Gate threshold (CI lower bound ≥ 0.85)**: PASS

Interpretation: with 95% probability under non-parametric resampling, the true population macro-weighted F1 lies in [0.8861, 0.9355]. The point estimate 0.9112 passes the design-D3 gate of 0.85; the CI's lower bound also passes — i.e., we can claim the gate result is statistically robust to triple-level sampling noise.