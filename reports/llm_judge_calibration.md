# LLM-Judge Calibration — Phase 2 §12.2

Reliability diagrams + isotonic-calibration fit for the two CoT backends on the 588-triple human concordance set. Reads the per-call records dumped by `validate --records-out`. ECE is the standard expected-calibration-error metric: lower is better-calibrated.

> **Methodological caveat (added 2026-05-22 after external review):**
> the isotonic mapping is *fit and evaluated on the same 588 triples*.
> The post-fit ECE values below (~0.003 for mini-cot, ~0.000 for
> Together-cot) are therefore **in-sample** estimates and represent the
> upper bound on calibration quality, not held-out generalisation.
> Pool-adjacent-violators trivially achieves near-zero ECE on its
> training set by linear-interpolating between observed bins; the true
> downstream calibration quality on novel (sentence, abstract) pairs is
> likely worse. A defensive estimate would run k-fold CV (folds at
> qa_id boundaries to avoid topical leakage) and report the
> cross-validated ECE; that work is deferred. The raw ECE numbers
> (0.1136 mini, 0.0961 Together) are unaffected by this caveat — they
> measure the *uncalibrated* model, where train-test split is moot.


## openai-gpt-4o-mini-cot

- Triples: 588
- Overall accuracy: 0.8571
- Mean raw confidence: 0.8459
- **ECE (raw)**: 0.1136
- **ECE (after isotonic fit)**: 0.0032

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
- **ECE (after isotonic fit)**: 0.0000

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

If ECE (after isotonic) is ≪ ECE (raw), the fit recovered meaningful calibration structure. If they are similar, the raw confidences are already approximately well-calibrated — common for modern instruction-tuned LLMs on simple binary-ish tasks.