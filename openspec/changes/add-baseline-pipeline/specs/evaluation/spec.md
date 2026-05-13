## ADDED Requirements

### Requirement: Local replication of the official Strict and Relaxed metrics
The system SHALL provide an evaluation script that, given a submission JSONL and a qrels file in the official BioGEN format, computes per-class precision, recall, and F1 under both the Strict setting (positives = `Dsup` for support; `Dcon` for contradict) and the Relaxed setting (positives = `Dsup ∪ Dpsup` for support; `Dcon` for contradict).

#### Scenario: Evaluation produces both settings in one run
- **WHEN** the operator runs `python -m trec_biogen.eval --submission <s> --qrels <q>`
- **THEN** the script emits a single JSON report containing six numbers (precision, recall, F1) × two classes (support, contradict) × two settings (strict, relaxed)

#### Scenario: Soft recall is binary per sentence per class
- **WHEN** computing recall for a (sentence, class) pair
- **THEN** the contribution is 1 if at least one predicted PMID matches a positive in qrels, else 0

### Requirement: Evaluation report includes leaderboard comparison rows
The evaluation output SHALL include a Markdown table that places the current run alongside the published 2025 baseline (Supports F1 44.34, Contradicts F1 4.67), the top support system (CLaC, 67.74), and the top contradiction system (InfoLab, 14.15), so deltas are visible at a glance.

#### Scenario: Markdown table is written next to the JSON report
- **WHEN** evaluation completes
- **THEN** a `report.md` is written in the same directory containing the comparison table with the current run's row highlighted

### Requirement: Evaluation supports both 2024 and 2025 qrels
The evaluation script SHALL accept any qrels file in the official format and produce identical metric definitions regardless of which year's qrels it processes. Reports SHALL record which qrels file was used.

#### Scenario: 2024 qrels run
- **WHEN** the script is invoked with `--qrels data/qrels/biogen_2024.jsonl`
- **THEN** the report's metadata records the qrels filename and the metric values are computed identically to the 2025 case

### Requirement: Baseline reproduction gate
The system SHALL provide a single command that runs the unmodified starter-kit baseline against the 2025 qrels and asserts that the resulting Supports F1 and Contradicts F1 are within ±2 absolute points of the published baseline (44.34 and 4.67). If either deviates by more than 2, the command SHALL exit non-zero.

#### Scenario: Local baseline matches published baseline
- **WHEN** the operator runs `make baseline-check` (or equivalent)
- **THEN** the command runs the baseline pipeline and exits 0 only if Supports F1 ∈ [42.34, 46.34] AND Contradicts F1 ∈ [2.67, 6.67]

#### Scenario: Local baseline diverges
- **WHEN** the local replication produces F1 outside the tolerance
- **THEN** the command exits non-zero and the report.md flags the divergence with the absolute deltas

### Requirement: Phase-1 acceptance thresholds
The evaluation report SHALL flag whether the current run meets the Phase-1 acceptance thresholds defined in `proposal.md`: Supports F1 ≥ 60 AND Contradicts F1 ≥ 10 under the Strict setting against the 2025 qrels.

#### Scenario: Phase-1 targets met
- **WHEN** Supports F1 ≥ 60 AND Contradicts F1 ≥ 10 (strict, 2025 qrels)
- **THEN** the report includes a `phase1_pass: true` field

#### Scenario: Phase-1 targets not met
- **WHEN** either threshold is missed
- **THEN** the report includes `phase1_pass: false` and the per-class deltas to the threshold
