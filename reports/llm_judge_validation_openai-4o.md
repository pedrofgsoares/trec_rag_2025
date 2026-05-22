# LLM-Judge Concordance Validation

- Backend: `openai-gpt-4o`
- Qrels: `data/qrels/biogen2025_taskA_qrels.jsonl`
- Threshold (macro weighted F1): `0.85`
- Triples scored: 588
- **Macro weighted F1: 0.7443** (FAIL)

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supports | 0.9748 | 0.6339 | 0.7682 | 549 |
| Contradicts | 0.7333 | 0.2821 | 0.4074 | 39 |
| Neutral | 0.0000 | 0.0000 | 0.0000 | 0 |
| Not relevant | 0.0000 | 0.0000 | 0.0000 | 0 |

## Confusion matrix

Rows: human label. Columns: judge label.

| | Supports | Contradicts | Neutral | Not relevant |
|---|---|---|---|---|
| Supports | 348 | 4 | 167 | 30 |
| Contradicts | 9 | 11 | 15 | 4 |
| Neutral | 0 | 0 | 0 | 0 |
| Not relevant | 0 | 0 | 0 | 0 |
