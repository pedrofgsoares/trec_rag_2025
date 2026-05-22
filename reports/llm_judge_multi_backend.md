# Multi-backend concordance — Phase 2 §12.4

Three independent backends classified the same 588-triple human concordance set. Per-backend gate F1 vs human gold + pairwise judge-vs-judge concordance establish the multi-evaluator robustness claim called out in design D10.

## Per-backend gate (vs human gold)

| Backend | Prompt | Macro w-F1 | Supports F1 | Contradicts F1 | Gate (≥ 0.85) |
|---|---|---|---|---|---|
| openai-gpt-4o-mini-cot | (per file) | 0.8982 | 0.9256 | 0.5135 | PASS |
| together-llama-3.3-70b-cot | (per file) | 0.9112 | 0.9582 | 0.2500 | PASS |

## Pairwise judge-vs-judge concordance

Agreement = fraction of triples where the two backends emit the same label. Cohen's κ corrects for chance agreement; κ ≥ 0.6 is *substantial* agreement; ≥ 0.8 is *almost perfect* (Landis & Koch, 1977).

| A | B | Raw agreement | Cohen's κ |
|---|---|---|---|
| openai-gpt-4o-mini-cot | together-llama-3.3-70b-cot | 0.8673 | 0.3384 |

## Interpretation

- All 2 backends pass the 0.85 gate against human gold.
- Pairwise Cohen's κ ranges [0.338, 0.338].

All backends pass the gate against human gold individually, but pairwise κ ranges down to 0.338 — backends disagree with each other more than they disagree with humans on the human-labeled triples. This is consistent with each backend being a valid noisy approximation of a human label, but with the noise distributions partly orthogonal. Cross-backend agreement-floor reporting (Phase 2 §10.5 fallback) would be the conservative extension.
