# LLM-Judge Concordance Validation

- Backend: `openai-gpt-4o-mini`
- Qrels: `data/qrels/biogen2025_taskA_qrels.jsonl`
- Threshold (macro weighted F1): `0.85`
- Triples scored: 588
- **Macro weighted F1: 0.7497** (FAIL)

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supports | 0.9750 | 0.6393 | 0.7723 | 549 |
| Contradicts | 0.4571 | 0.4103 | 0.4324 | 39 |
| Neutral | 0.0000 | 0.0000 | 0.0000 | 0 |
| Not relevant | 0.0000 | 0.0000 | 0.0000 | 0 |

## Confusion matrix

Rows: human label. Columns: judge label.

| | Supports | Contradicts | Neutral | Not relevant |
|---|---|---|---|---|
| Supports | 351 | 19 | 171 | 8 |
| Contradicts | 9 | 16 | 14 | 0 |
| Neutral | 0 | 0 | 0 | 0 |
| Not relevant | 0 | 0 | 0 | 0 |
