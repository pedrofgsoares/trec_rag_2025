## ADDED Requirements

### Requirement: Dual-pool scoring via --qrels-pool flag
The evaluation script SHALL accept a `--qrels-pool={official,expanded}` flag that selects which qrels file to score against. The default value SHALL remain `official` so that any historical invocation reproduces the §6.5 baseline numbers byte-for-byte. When `--qrels-pool=expanded` is set, the script SHALL load `data/qrels/biogen2025_taskA_qrels_expanded.jsonl` and apply the same metric definitions.

#### Scenario: Default pool is official
- **WHEN** `python -m trec_biogen.eval.metrics --submission <s>` is invoked without `--qrels-pool`
- **THEN** the script reads `data/qrels/biogen2025_taskA_qrels.jsonl` and produces the canonical official-pool report

#### Scenario: Expanded pool is opt-in
- **WHEN** `--qrels-pool=expanded` is set and `data/qrels/biogen2025_taskA_qrels_expanded.jsonl` does not exist
- **THEN** the script exits non-zero with an error message instructing the operator to run the LLM-judge rejudge CLI first

#### Scenario: Source filter restricts to human positives
- **WHEN** `--source=human` is set against the expanded qrels
- **THEN** only records with `source: "human"` participate as positives, reproducing the official-pool numbers exactly from the expanded file

### Requirement: Per-variant Phase 2 summary report
The system SHALL provide a reporting command that consumes all `runs/` directories tagged as Phase 2 variants and writes `reports/phase2_summary.md` with one row per variant. Each row SHALL include: variant name, F1 on official pool (support, contradict), F1 on expanded pool (support, contradict), Δ official→expanded (per class), wall-clock seconds, peak VRAM GB, and judge cost USD (if applicable).

#### Scenario: Summary is regenerated after every new variant
- **WHEN** the operator runs `python -m trec_biogen.eval.phase2_summary`
- **THEN** the command scans `runs/`, identifies every directory whose `metadata.yaml` contains a `phase2_variant` key, and rewrites `reports/phase2_summary.md` with one row per such variant ordered by variant name

#### Scenario: Δ column quantifies pool-bias contribution
- **WHEN** the summary row for a variant is rendered
- **THEN** the Δ column for each class equals `F1_expanded - F1_official` and is rendered with explicit sign (e.g., `+2.34`, `-0.15`)

### Requirement: Cost / wall-clock / VRAM captured per run for Pareto plotting
For every Phase 2 pipeline run, `metadata.yaml` SHALL include `phase2_variant: <name>`, `wall_clock_seconds_total: int`, `vram_peak_gb_total: float`, and `judge_cost_usd: float` (zero for runs that do not invoke the judge). These fields enable a Pareto-frontier visualisation across variants.

#### Scenario: Variant tag enables summary filtering
- **WHEN** any Phase 2 variant is invoked via its `configs/run/phase2_<name>.yaml`
- **THEN** the resolved Hydra config sets `phase2_variant: <name>` and the orchestrator writes that value into `metadata.yaml`

#### Scenario: Pipeline runs without LLM judge record zero cost
- **WHEN** a Phase 2 run completes without invoking the LLM-judge module
- **THEN** `judge_cost_usd: 0.0` and `judge_token_breakdown: {input_tokens: 0, output_tokens: 0, cache_hit_rate: 0.0}` appear in `metadata.yaml`

## MODIFIED Requirements

### Requirement: Local replication of the official Strict and Relaxed metrics
The system SHALL provide an evaluation script that, given a submission JSONL and a qrels file in the official BioGEN format, computes per-class precision, recall, and F1 under both the Strict setting (positives = `Dsup` for support; `Dcon` for contradict) and the Relaxed setting (positives = `Dsup ∪ Dpsup` for support; `Dcon` for contradict). The script SHALL apply the published BioGEN 2025 macro-averaging methodology: per-cell F1 averaged across all `(qa_id, sentence_id, class)` cells present in the submission, with cells lacking positives in the qrels contributing F1 = 0. The script SHALL accept any qrels file conforming to the schema (official or LLM-expanded; see `--qrels-pool` flag).

#### Scenario: Evaluation produces both settings in one run
- **WHEN** the operator runs `python -m trec_biogen.eval.metrics --submission <s> --qrels <q>`
- **THEN** the script emits a single JSON report containing six numbers (precision, recall, F1) × two classes (support, contradict) × two settings (strict, relaxed)

#### Scenario: Soft recall is binary per sentence per class
- **WHEN** computing recall for a (sentence, class) pair
- **THEN** the contribution is 1 if at least one predicted PMID matches a positive in qrels, else 0

#### Scenario: Unjudged cells contribute F1 = 0
- **WHEN** a `(qa_id, sentence_id, class)` cell exists in the submission but has zero positives in the qrels
- **THEN** that cell contributes F1 = 0 to the macro average (matching the published methodology)

### Requirement: Evaluation report includes leaderboard comparison rows
The evaluation output SHALL include a Markdown table that places the current run alongside the published 2025 baseline (Supports F1 44.34, Contradicts F1 4.67), the top support system (CLaC, 67.74), the top contradiction system (InfoLab, 14.15), AND the dual-pool numbers (official and expanded) so deltas vs both anchor and Phase 1 are visible at a glance.

#### Scenario: Markdown table is written next to the JSON report
- **WHEN** evaluation completes
- **THEN** a `report.md` is written in the same directory containing the comparison table with the current run's row highlighted

#### Scenario: Dual-pool row when expanded qrels are available
- **WHEN** `--qrels-pool=expanded` is used AND the expanded qrels file is present
- **THEN** the comparison table includes one row for the current run's official-pool F1 and a separate row for the expanded-pool F1 with the Δ explicit
