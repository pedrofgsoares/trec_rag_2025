# LLM-Judge Concordance Validation

- Backend: `openai-gpt-4o-mini`
- Qrels: `data/qrels/biogen2025_taskA_qrels.jsonl`
- Threshold (macro weighted F1): `0.85`
- Triples scored: 588
- **Macro weighted F1: 0.8944** (PASS)

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supports | 0.9681 | 0.8834 | 0.9238 | 549 |
| Contradicts | 0.5000 | 0.4615 | 0.4800 | 39 |
| Neutral | 0.0000 | 0.0000 | 0.0000 | 0 |
| Not relevant | 0.0000 | 0.0000 | 0.0000 | 0 |

## Confusion matrix

Rows: human label. Columns: judge label.

| | Supports | Contradicts | Neutral | Not relevant |
|---|---|---|---|---|
| Supports | 485 | 18 | 45 | 1 |
| Contradicts | 16 | 18 | 5 | 0 |
| Neutral | 0 | 0 | 0 | 0 |
| Not relevant | 0 | 0 | 0 | 0 |
