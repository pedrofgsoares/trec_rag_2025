# LLM-Judge Concordance Validation

- Backend: `together-llama-3.3-70b`
- Qrels: `data/qrels/biogen2025_taskA_qrels.jsonl`
- Threshold (macro weighted F1): `0.85`
- Triples scored: 588
- **Macro weighted F1: 0.9112** (PASS)

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supports | 0.9564 | 0.9599 | 0.9582 | 549 |
| Contradicts | 0.6667 | 0.1538 | 0.2500 | 39 |
| Neutral | 0.0000 | 0.0000 | 0.0000 | 0 |
| Not relevant | 0.0000 | 0.0000 | 0.0000 | 0 |

## Confusion matrix

Rows: human label. Columns: judge label.

| | Supports | Contradicts | Neutral | Not relevant |
|---|---|---|---|---|
| Supports | 527 | 3 | 13 | 6 |
| Contradicts | 24 | 6 | 9 | 0 |
| Neutral | 0 | 0 | 0 | 0 |
| Not relevant | 0 | 0 | 0 | 0 |
