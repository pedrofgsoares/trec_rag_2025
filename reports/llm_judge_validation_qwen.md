# LLM-Judge Concordance Validation

- Backend: `hf-qwen-2.5-72b`
- Qrels: `data/qrels/biogen2025_taskA_qrels.jsonl`
- Threshold (macro weighted F1): `0.85`
- Per-class F1 floor (every class with non-zero gold support): `0.05` (defensive: rules out degenerate single-class classifiers; see `passes()` docstring)
- Triples scored: 588
- **Macro weighted F1: 0.8980** (PASS)

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supports | 0.9519 | 0.9362 | 0.9440 | 549 |
| Contradicts | 0.6667 | 0.1538 | 0.2500 | 39 |
| Neutral | 0.0000 | 0.0000 | 0.0000 | 0 |
| Not relevant | 0.0000 | 0.0000 | 0.0000 | 0 |

## Confusion matrix

Rows: human label. Columns: judge label.

| | Supports | Contradicts | Neutral | Not relevant |
|---|---|---|---|---|
| Supports | 514 | 3 | 30 | 2 |
| Contradicts | 26 | 6 | 6 | 1 |
| Neutral | 0 | 0 | 0 | 0 |
| Not relevant | 0 | 0 | 0 | 0 |
