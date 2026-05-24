## Purpose

Sentence-level macro-F1 evaluation for BioGEN 2025 Task A submissions, with dual-pool scoring (official human qrels vs LLM-augmented expanded pools) and reproducibility anchors.
## Requirements
### Requirement: Dual-pool scoring via --qrels-pool flag
The evaluation script SHALL accept a `--qrels-pool={official,expanded,intersection,intersection-3way}` flag that selects which qrels file to score against. The default value SHALL remain `official` so that any historical invocation reproduces the §6.5 baseline numbers byte-for-byte. When `--qrels-pool=expanded` is set, the script SHALL load `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`. When `--qrels-pool=intersection` is set, the script SHALL load `data/qrels/biogen2025_taskA_qrels_intersection.jsonl` (the two-judge intersection-on-contradicts pool emitted by the `llm-judge` capability). When `--qrels-pool=intersection-3way` is set, the script SHALL load `data/qrels/biogen2025_taskA_qrels_intersection_3way.jsonl` (the three-judge intersection-on-contradicts pool: Supports from the canonical mini-cot, Contradicts requiring unanimous agreement across mini-cot, Llama-3.3-70B-cot, and Qwen2.5-72B-cot). All four pools SHALL use identical metric definitions.

#### Scenario: Default pool is official
- **WHEN** `python -m trec_biogen.eval.metrics --submission <s>` is invoked without `--qrels-pool`
- **THEN** the script reads `data/qrels/biogen2025_taskA_qrels.jsonl` and produces the canonical official-pool report

#### Scenario: Expanded pool is opt-in
- **WHEN** `--qrels-pool=expanded` is set and `data/qrels/biogen2025_taskA_qrels_expanded.jsonl` does not exist
- **THEN** the script exits non-zero with an error message instructing the operator to run the LLM-judge rejudge CLI first

#### Scenario: Intersection pool is opt-in and requires both judges' files
- **WHEN** `--qrels-pool=intersection` is set and either `biogen2025_taskA_qrels_expanded.jsonl` or `biogen2025_taskA_qrels_expanded_together.jsonl` is missing
- **THEN** the script exits non-zero naming the missing file and pointing to the `llm-judge intersection.emit_intersection_pool` helper

#### Scenario: Three-way intersection pool is opt-in and requires all three judges' files
- **WHEN** `--qrels-pool=intersection-3way` is set and any of `biogen2025_taskA_qrels_expanded.jsonl`, `biogen2025_taskA_qrels_expanded_together.jsonl`, or `biogen2025_taskA_qrels_expanded_qwen.jsonl` is missing
- **THEN** the script exits non-zero naming the missing file(s) and pointing to the `llm-judge intersection.emit_intersection_pool` helper with `records_paths=[mini, llama, qwen]`

#### Scenario: Three-way intersection pool reproducibility anchor
- **WHEN** `--qrels-pool=intersection-3way --source=human` is set
- **THEN** the script recovers the published 44.34 Supports F1 byte-for-byte (the human positives are pass-through-identical across all four pool variants)

#### Scenario: Source filter restricts to human positives
- **WHEN** `--source=human` is set against any of the expanded, intersection, or intersection-3way qrels
- **THEN** only records with `source: "human"` participate as positives, reproducing the official-pool numbers exactly from any of the three augmented files

### Requirement: Topic-level F1 column in the Phase 2 summary
The `eval/phase2_summary.py` output SHALL be extended with an optional per-topic breakdown view, enabled via `--by-topic`. When enabled, the report SHALL list per-`qa_id` F1 deltas vs the starter baseline across all listed runs and pools. The headline cell-level macro table SHALL remain the default view.

#### Scenario: Default view is unchanged
- **WHEN** `python -m trec_biogen.eval.phase2_summary` is invoked without `--by-topic`
- **THEN** the output is byte-for-byte identical to the Phase 2 baseline summary (one row per variant, three pools, cell-level macro)

#### Scenario: --by-topic adds a second table
- **WHEN** `python -m trec_biogen.eval.phase2_summary --by-topic` is invoked
- **THEN** the report carries the default cell-level table followed by a second table with one row per `qa_id` and columns per `(variant, pool)`, scored using the per-topic aggregation

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

### Requirement: Cost / wall-clock / VRAM captured per run for Pareto plotting
For every Phase 2 pipeline run, `metadata.yaml` SHALL include `phase2_variant: <name>`, `wall_clock_seconds_total: int`, `vram_peak_gb_total: float`, and `judge_cost_usd: float` (zero for runs that do not invoke the judge). These fields enable a Pareto-frontier visualisation across variants.

#### Scenario: Variant tag enables summary filtering
- **WHEN** any Phase 2 variant is invoked via its `configs/run/phase2_<name>.yaml`
- **THEN** the resolved Hydra config sets `phase2_variant: <name>` and the orchestrator writes that value into `metadata.yaml`

#### Scenario: Pipeline runs without LLM judge record zero cost
- **WHEN** a Phase 2 run completes without invoking the LLM-judge module
- **THEN** `judge_cost_usd: 0.0` and `judge_token_breakdown: {input_tokens: 0, output_tokens: 0, cache_hit_rate: 0.0}` appear in `metadata.yaml`

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

### Requirement: Krippendorff α helper for multi-judge agreement
The evaluation module SHALL provide a `metrics.krippendorff_alpha(labels_per_coder, *, classes, missing_marker=None)` function that takes a list of per-coder label sequences (one list per coder, all of equal length and aligned by unit index) plus the categorical label space, and returns Krippendorff's α for nominal data using the **Krippendorff 2011** standard formulation `α = 1 - D_observed / D_expected` with the disagreement function `δ(c1, c2) = 0 if c1 == c2 else 1`. Per-unit numerator contribution is `(m_u² − Σ_c n_uc²) / (m_u − 1)` (units with `m_u < 2` contribute zero); `D_obs` divides the sum of per-unit contributions by the total number of non-missing labels `N`. The implementation SHALL handle missing values via the `missing_marker` parameter so published reference fixtures (which contain missing data) can be reproduced; in the primary use case (every backend judges every triple) `missing_marker` is left at its default and the formula reduces to the simple equal-m formulation. The function SHALL be pure (no I/O), SHALL be deterministic, and SHALL match the K2011-formula value `α ≈ 0.7520` on the Hayes & Krippendorff (2007) Table 1 fixture to within `1e-3`.

*Implementation note*: the Hayes & Krippendorff (2007) paper itself reports `α = 0.7434` for the Table 1 fixture, but that value comes from the SPSS KALPHA macro (which uses a slightly different coincidence-matrix normalisation per Krippendorff 2004). Modern implementations following Krippendorff (2011) — including this one and the `krippendorff` PyPI package — report `0.7520` on the same fixture; the ~0.01 inter-implementation variance is documented in the field and is not a bug. We anchor to the K2011 value because that is the *current* canonical formula.

#### Scenario: Reference fixture matches the K2011 formula's value
- **WHEN** `krippendorff_alpha` is called with the labels-per-coder matrix from Hayes & Krippendorff (2007) Table 1 (the `"*"` cells passed as the missing marker) and `classes=("1","2","3","4","5"), missing_marker="*"`
- **THEN** the returned α equals `0.7520` to within `1e-3` (the K2011-standard formula value; see implementation note above re: 0.7434 vs 0.7520)

#### Scenario: Three-judge α reported per-class and overall
- **WHEN** the judge-intersection analysis is regenerated with three backends' labels for the 588-triple gold set
- **THEN** `reports/judge_intersection_analysis.md` reports an overall α plus per-class α (one α computed with `classes=("Supports","Contradicts","Neutral")` restricted to units whose human label is in that class)

#### Scenario: Mismatched sequence lengths
- **WHEN** the input lists of labels have different lengths
- **THEN** the function raises `ValueError` before doing any computation

