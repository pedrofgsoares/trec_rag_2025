## ADDED Requirements

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

## MODIFIED Requirements

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
