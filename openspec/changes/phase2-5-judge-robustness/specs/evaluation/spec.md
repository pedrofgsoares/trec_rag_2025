## ADDED Requirements

### Requirement: Topic-level F1 column in the Phase 2 summary
The `eval/phase2_summary.py` output SHALL be extended with an optional per-topic breakdown view, enabled via `--by-topic`. When enabled, the report SHALL list per-`qa_id` F1 deltas vs the starter baseline across all listed runs and pools. The headline cell-level macro table SHALL remain the default view.

#### Scenario: Default view is unchanged
- **WHEN** `python -m trec_biogen.eval.phase2_summary` is invoked without `--by-topic`
- **THEN** the output is byte-for-byte identical to the Phase 2 baseline summary (one row per variant, three pools, cell-level macro)

#### Scenario: --by-topic adds a second table
- **WHEN** `python -m trec_biogen.eval.phase2_summary --by-topic` is invoked
- **THEN** the report carries the default cell-level table followed by a second table with one row per `qa_id` and columns per `(variant, pool)`, scored using the per-topic aggregation

## MODIFIED Requirements

### Requirement: Dual-pool scoring via --qrels-pool flag
The evaluation script SHALL accept a `--qrels-pool={official,expanded,intersection}` flag that selects which qrels file to score against. The default value SHALL remain `official` so that any historical invocation reproduces the §6.5 baseline numbers byte-for-byte. When `--qrels-pool=expanded` is set, the script SHALL load `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`. When `--qrels-pool=intersection` is set, the script SHALL load `data/qrels/biogen2025_taskA_qrels_intersection.jsonl` (the two-judge intersection-on-contradicts pool emitted by the `llm-judge` capability). All three pools SHALL use identical metric definitions.

#### Scenario: Default pool is official
- **WHEN** `python -m trec_biogen.eval.metrics --submission <s>` is invoked without `--qrels-pool`
- **THEN** the script reads `data/qrels/biogen2025_taskA_qrels.jsonl` and produces the canonical official-pool report

#### Scenario: Expanded pool is opt-in
- **WHEN** `--qrels-pool=expanded` is set and `data/qrels/biogen2025_taskA_qrels_expanded.jsonl` does not exist
- **THEN** the script exits non-zero with an error message instructing the operator to run the LLM-judge rejudge CLI first

#### Scenario: Intersection pool is opt-in and requires both judges' files
- **WHEN** `--qrels-pool=intersection` is set and either `biogen2025_taskA_qrels_expanded.jsonl` or `biogen2025_taskA_qrels_expanded_together.jsonl` is missing
- **THEN** the script exits non-zero naming the missing file and pointing to the `llm-judge intersection.emit_intersection_pool` helper

#### Scenario: Source filter restricts to human positives
- **WHEN** `--source=human` is set against the expanded or intersection qrels
- **THEN** only records with `source: "human"` participate as positives, reproducing the official-pool numbers exactly from either expanded file

### Requirement: Per-variant Phase 2 summary report
The system SHALL provide a reporting command that consumes all `runs/` directories tagged as Phase 2 variants and writes `reports/phase2_summary.md` with one row per variant. Each row SHALL include: variant name, F1 on official pool (support, contradict), F1 on expanded pool (support, contradict), F1 on intersection pool (support, contradict — contradict numbers use the two-judge intersection; support numbers default to the canonical expanded), Δ official→expanded (per class), wall-clock seconds, peak VRAM GB, and judge cost USD (if applicable). Pool columns where the underlying file is absent SHALL render as `n/a` rather than blocking the report.

#### Scenario: Summary is regenerated after every new variant
- **WHEN** the operator runs `python -m trec_biogen.eval.phase2_summary`
- **THEN** the command scans `runs/`, identifies every directory whose `metadata.yaml` contains a `phase2_variant` key, and rewrites `reports/phase2_summary.md` with one row per such variant ordered by variant name

#### Scenario: Δ column quantifies pool-bias contribution
- **WHEN** the summary row for a variant is rendered
- **THEN** the Δ column for each class equals `F1_expanded - F1_official` and is rendered with explicit sign (e.g., `+2.34`, `-0.15`)

#### Scenario: Intersection pool unavailable degrades gracefully
- **WHEN** the intersection qrels file is missing at summary regeneration time
- **THEN** the intersection-pool columns render as `n/a` for every row and the summary still emits successfully so the existing Phase 2 outputs are not broken
