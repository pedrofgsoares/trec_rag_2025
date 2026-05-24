## Purpose

Per-`qa_id` F1 analysis of pipeline variants, plus the `per_topic_diff` script for inspecting concrete PMID-level disagreements between two runs on a chosen topic.

## Requirements

### Requirement: Topic-level F1 aggregation from per-cell metrics
The evaluation module SHALL provide a `per_topic_f1(run_dir, *, pool="intersection")` helper that, given a completed run directory and a qrels-pool selector, returns a mapping `qa_id -> {support: {P, R, F1, n_cells}, contradict: {P, R, F1, n_cells}}`. The per-cell numbers SHALL be re-derived from the run's cached `task_a_output.json` and the chosen qrels pool; the aggregation SHALL be the arithmetic mean of cell-level F1 across the cells belonging to that `qa_id`.

#### Scenario: Topic aggregation from cached run
- **WHEN** `per_topic_f1("runs/20260516-134227-phase1_baseline", pool="intersection")` is called
- **THEN** the helper returns one entry per `qa_id` present in the run's submission, with the per-class macro derived from the intersection-pool qrels

#### Scenario: Default pool matches headline
- **WHEN** the `pool` argument is omitted
- **THEN** the default is `"intersection"` so the qualitative analysis defaults to the conservative pool; passing `pool="expanded"` or `pool="official"` selects the other two pools without code changes

#### Scenario: Empty cells in a topic
- **WHEN** a `qa_id` has cells with submissions but no positives in the chosen pool
- **THEN** those cells contribute F1 = 0 to the topic's mean (same `unjudged_as_zero=True` convention as the cell-level macro)

### Requirement: Cross-run topic-level diff CLI
The system SHALL provide a `scripts/per_topic_diff.py` CLI that, given two run directories and a `qa_id`, prints a side-by-side comparison of the PMIDs the two runs emitted for that topic, the set differences in each class, and for each PMID in `A \ B` and `B \ A` the LLM-judge label and confidence drawn from the chosen qrels pool (default: intersection).

#### Scenario: Diff prints set differences with judge attribution
- **WHEN** the operator runs `python scripts/per_topic_diff.py --a runs/<phase1> --b runs/<starter> --qa-id 116`
- **THEN** the script prints, for both classes: the PMIDs in A but not B, the PMIDs in B but not A, each PMID's class label and confidence from the intersection qrels (or `<unjudged>` if absent), and for each PMID where rejudge records exist a one-line excerpt of the judge's reasoning chain

#### Scenario: Missing qa_id is reported explicitly
- **WHEN** the requested `qa_id` is not present in one of the runs' submissions
- **THEN** the CLI exits non-zero with a clear message naming the run and the missing identifier

### Requirement: Mechanical topic selection for qualitative analysis
The system SHALL provide a routine that, given a target run and an anchor run (default: `starter_baseline_*`), returns the three topics with: (a) the largest positive `target.topic_f1 - anchor.topic_f1`, (b) the value closest to zero, and (c) the largest negative. The selection routine SHALL be deterministic and the chosen topics SHALL be recorded in the qualitative-analysis report so a reader can verify the selection was not cherry-picked.

#### Scenario: Selection on the intersection pool
- **WHEN** the routine is called with `target=runs/<phase1>` and `anchor=runs/<starter>` on `pool="intersection"`
- **THEN** the three selected `qa_id`s are printed with their per-topic F1 deltas, and the report includes the full sorted list as an appendix so the choice is auditable

#### Scenario: Tie-breaking is deterministic
- **WHEN** two topics have identical deltas
- **THEN** the routine breaks ties by ascending `qa_id` (integer order) so re-running the selection yields the same three topics