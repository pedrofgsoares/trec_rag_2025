# Bootstrap 95% CI — openai-gpt-4o-mini (--prompt cot)

- Sample: 588 (gold, pred) pairs from the §2.15 concordance run
- Bootstrap iterations: 1000, seed: 0
- Gate threshold: 0.85

## Per-class point estimates

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supports | 0.9719 | 0.8834 | 0.9256 | 549 |
| Contradicts | 0.5429 | 0.4872 | 0.5135 | 39 |
| Neutral | 0.0000 | 0.0000 | 0.0000 | 0 |
| Not relevant | 0.0000 | 0.0000 | 0.0000 | 0 |

## Macro-weighted F1 with bootstrap 95% CI

- **Point estimate**: 0.8982 (PASS)
- **95% CI**: [0.8776, 0.9196]  (width 0.0419)
- **Gate threshold (CI lower bound ≥ 0.85)**: PASS

Interpretation: with 95% probability under non-parametric resampling, the true population macro-weighted F1 lies in [0.8776, 0.9196]. The point estimate 0.8982 passes the design-D3 gate of 0.85; the CI's lower bound also passes — i.e., we can claim the gate result is statistically robust to triple-level sampling noise.