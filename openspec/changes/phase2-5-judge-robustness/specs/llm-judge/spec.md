## ADDED Requirements

### Requirement: Multi-judge rejudge over a fixed candidate set
The judge module SHALL support running the rejudge CLI more than once over the same candidate-triple set (same `--submission`, same `--mode expand-pool`) with different `--backend` values, each emitting to its own qrels file under `data/qrels/biogen2025_taskA_qrels_expanded_<backend-tag>.jsonl` (e.g. `expanded.jsonl` for the canonical mini-cot default; `expanded_together.jsonl` for Together-Llama-3.3-70B). Each file SHALL preserve the same JSONL schema as the canonical expanded qrels and carry `source: "llm-<backend-tag>"` per LLM record.

#### Scenario: Second-backend rejudge does not mutate the first
- **WHEN** the operator runs `judge.rejudge expand-pool --backend together --out data/qrels/biogen2025_taskA_qrels_expanded_together.jsonl` against the same candidate set as a prior mini-cot rejudge
- **THEN** the canonical `expanded.jsonl` is unchanged, the new `expanded_together.jsonl` is written, and both can be loaded independently by the same parser

#### Scenario: Backend tag inferred from --backend
- **WHEN** the `--out` flag is omitted on a multi-judge rejudge invocation
- **THEN** the output filename is derived as `expanded_<backend-tag>.jsonl` using the backend's canonical short name (`together`, `openai-mini`, `openai-4o`) so the per-backend artefact does not need explicit naming

### Requirement: Two-judge intersection pool emitter for the Contradicts class
The judge module SHALL provide an `intersection.emit_intersection_pool(records_a, records_b, *, human_qrels, out_path)` helper that, given two backend-tagged expanded qrels files (each a superset of the human qrels), emits a third qrels file containing: (a) every human-labelled record copied verbatim; (b) every Supports record from the canonical (first-arg) judge — Supports are not intersected because §12.4 shows them robust to judge choice; (c) every Contradicts record present in BOTH judges' files with matching `(qa_id, sentence_id, pmid, class)`. The emitted file SHALL be valid against the canonical qrels parser and SHALL carry a sidecar `<out_path>.meta.json` with: SHA256 of both input files, the intersection rule applied per class, the resulting per-class positive counts, and an `incomplete: true` flag if either input was marked incomplete.

#### Scenario: Contradicts intersected, Supports passed through
- **WHEN** two backends' expanded qrels are passed to the emitter
- **THEN** the output's Contradicts positives are the set intersection by `(qa_id, sentence_id, pmid)`, the output's Supports positives are exactly the first-arg's Supports, and the human positives are bitwise-identical to the input human qrels

#### Scenario: Sidecar metadata is exhaustive
- **WHEN** the emitter completes
- **THEN** `<out_path>.meta.json` exists and contains the SHA256 of both input files, the timestamp, the per-class positive count before and after intersection, and the percentage of contradicts dropped relative to the union

#### Scenario: Either input incomplete propagates
- **WHEN** either input qrels' sidecar metadata carries `incomplete: true`
- **THEN** the emitted intersection pool's sidecar also carries `incomplete: true` and the report writer downweights its conclusions accordingly
