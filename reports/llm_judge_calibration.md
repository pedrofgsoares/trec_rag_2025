# LLM-Judge Calibration — Phase 2 §12.2 / Phase 2.6 §1

Reliability diagrams + isotonic-calibration fit for the CoT backends on the 588-triple human concordance set. Reads the per-call records dumped by `validate --records-out`. ECE is the standard expected-calibration-error metric: lower is better-calibrated.

> **Phase 2.6 update (2026-05-23):** the post-isotonic ECE is now reported as a **held-out k=5 cross-validated mean** with folds split at `qa_id` boundaries (so the same topic never appears in both the PAV fit and the evaluation). This closes the in-sample caveat added in the Phase 2.5 code-review pass: PAV trivially achieves near-zero ECE on its training set, so the in-sample values reported in Phase 2 (~0.003 mini, ~0.000 Together) were upper bounds. The held-out numbers below are the defensible figure. Raw (uncalibrated) ECE is fit-free and matches the Phase 2 value byte-for-byte.


## openai-gpt-4o-mini-cot

- Triples: 588
- Overall accuracy: 0.8571
- Mean raw confidence: 0.8459
- **ECE (raw)**: 0.1136
- **ECE (after isotonic, in-sample)**: 0.0032 *(upper bound — pre-2026-05-23 number, kept for reference)*
- **ECE (after isotonic, k=5-fold held-out CV)**: 0.0476 ± 0.0225 *(defensible figure; folds split by `qa_id`)*
- Held-out fold sizes: [111, 157, 92, 70, 158]

### Reliability diagram (raw confidences)

| Bin (raw conf range) | n | mean pred | empirical acc | gap |
|---|---:|---:|---:|---:|
| [0.00, 0.10) | 0 | — | — | — |
| [0.10, 0.20) | 0 | — | — | — |
| [0.20, 0.30) | 0 | — | — | — |
| [0.30, 0.40) | 0 | — | — | — |
| [0.40, 0.50) | 0 | — | — | — |
| [0.50, 0.60) | 0 | — | — | — |
| [0.60, 0.70) | 14 | 0.6000 | 0.0000 | +0.6000 |
| [0.70, 0.80) | 51 | 0.7000 | 0.2745 | +0.4255 |
| [0.80, 0.90) | 247 | 0.8285 | 0.9433 | -0.1148 |
| [0.90, 1.00) | 276 | 0.9009 | 0.9312 | -0.0303 |

### ASCII reliability curve (raw vs empirical)

```
bin               n   pred    acc  curve
[0.60,0.70)   14  0.600  0.000  A·················P············
[0.70,0.80)   51  0.700  0.275  ········A············P·········
[0.80,0.90)  247  0.829  0.943  ·························P··A··
[0.90,1.00)  276  0.901  0.931  ···························PA··
                                          P = predicted prob (model)
                                          A = empirical accuracy
                                          X = aligned (P == A)
```

### Isotonic calibration fit

Pool-adjacent-violators (PAV) fit. 5 monotone blocks. Apply with `apply_isotonic(raw_confidence, mapping)`.

| raw conf ≤ | calibrated prob |
|---|---|
| 0.6000 | 0.0000 |
| 0.7000 | 0.2745 |
| 0.8000 | 0.9151 |
| 0.9000 | 0.9417 |
| 0.9500 | 1.0000 |

## together-llama-3.3-70b-cot

- Triples: 588
- Overall accuracy: 0.9065
- Mean raw confidence: 0.8532
- **ECE (raw)**: 0.0961
- **ECE (after isotonic, in-sample)**: 0.0000 *(upper bound — pre-2026-05-23 number, kept for reference)*
- **ECE (after isotonic, k=5-fold held-out CV)**: 0.0329 ± 0.0278 *(defensible figure; folds split by `qa_id`)*
- Held-out fold sizes: [111, 157, 92, 70, 158]

### Reliability diagram (raw confidences)

| Bin (raw conf range) | n | mean pred | empirical acc | gap |
|---|---:|---:|---:|---:|
| [0.00, 0.10) | 1 | 0.0000 | 0.0000 | +0.0000 |
| [0.10, 0.20) | 0 | — | — | — |
| [0.20, 0.30) | 0 | — | — | — |
| [0.30, 0.40) | 0 | — | — | — |
| [0.40, 0.50) | 0 | — | — | — |
| [0.50, 0.60) | 0 | — | — | — |
| [0.60, 0.70) | 4 | 0.6000 | 0.0000 | +0.6000 |
| [0.70, 0.80) | 16 | 0.7000 | 0.0625 | +0.6375 |
| [0.80, 0.90) | 223 | 0.8000 | 0.9058 | -0.1058 |
| [0.90, 1.00) | 344 | 0.9003 | 0.9593 | -0.0590 |

### ASCII reliability curve (raw vs empirical)

```
bin               n   pred    acc  curve
[0.00,0.10)    1  0.000  0.000  X······························
[0.60,0.70)    4  0.600  0.000  A·················P············
[0.70,0.80)   16  0.700  0.062  ··A··················P·········
[0.80,0.90)  223  0.800  0.906  ························P··A···
[0.90,1.00)  344  0.900  0.959  ···························P·A·
                                          P = predicted prob (model)
                                          A = empirical accuracy
                                          X = aligned (P == A)
```

### Isotonic calibration fit

Pool-adjacent-violators (PAV) fit. 6 monotone blocks. Apply with `apply_isotonic(raw_confidence, mapping)`.

| raw conf ≤ | calibrated prob |
|---|---|
| 0.0000 | 0.0000 |
| 0.6000 | 0.0000 |
| 0.7000 | 0.0625 |
| 0.8000 | 0.9058 |
| 0.9000 | 0.9592 |
| 0.9900 | 1.0000 |

## Interpretation

If ECE (raw) is *substantial* (≥ 0.05), the model's emitted confidences are not interchangeable with true probabilities; use the isotonic-calibrated values when applying a confidence threshold downstream (e.g., for two-backend agreement floors or selective rejudgment).

Compare the **in-sample** vs **held-out** post-isotonic ECE: the in-sample number is an upper bound (PAV interpolates between the exact bins it was fit on); the held-out number is what a deployed calibrator would actually achieve on novel `(sentence, abstract)` pairs from new topics. The gap quantifies how much of the apparent calibration quality is generalisation vs memorisation. With folds split at `qa_id` boundaries, the held-out estimate is conservative against the leakage mode that matters most for this task (same-topic PMIDs recurring across sentences).