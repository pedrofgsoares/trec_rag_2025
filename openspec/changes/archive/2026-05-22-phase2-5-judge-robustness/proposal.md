## Why

Phase 2 closed with a single-judge expanded pool (`gpt-4o-mini --prompt cot` over 5398 candidate triples → 4170 LLM positives), validated against the human pool but never tested for variant-ranking robustness across judges. The §12.4 multi-backend probe established that supports F1 is robust to judge choice but contradicts F1 carries meaningful judge-dependent variance (Cohen's κ = 0.34). The headline structural win of Phase 2 — `no_negex` beating Phase 1 on official Contradicts F1 by +2.13 pp — is exactly the kind of finding that needs a second-judge sanity check before being claimed in a report or paper. Separately, the Phase 2 commentary is quantitatively complete but qualitatively thin: there is no per-topic error analysis showing *where* and *why* internal variants beat or lose to the published baseline, which is a standard expectation for TREC-style reports.

## What Changes

- Run a second LLM-judge rejudge over the same 5398 candidate triples from §2.17 using `together-llama-3.3-70b --prompt cot` (the only backend that has cleared the 0.85 concordance gate alongside `gpt-4o-mini`). Persist results alongside the existing mini-cot records to enable diff analysis.
- Construct a **two-judge intersection pool** for the contradicts class: a triple counts as an LLM-positive Contradict only if *both* judges label it as such. Supports are not changed (already robust per §12.4); the intersection narrows only the contradict path.
- Extend the evaluation module with a third `--qrels-pool` value `intersection` (alongside `official` and `expanded`), wire it through `phase2_summary.py`, and re-score every existing run on the new pool.
- Add per-topic F1 breakdown to the evaluation output (already per-cell internally; needs a topic-level aggregation and a CLI for cross-run diffing).
- Build a `scripts/per_topic_diff.py` that compares two runs at the topic level and surfaces the PMIDs in `A \ B`, `B \ A`, with their LLM-judge verdicts and confidences.
- Select 3 representative topics (one large gain vs starter, one tie, one loss) and produce a qualitative error analysis section with 2 tables + 3 paragraphs for the report. Pool-source for the qualitative pass is the intersection pool (more conservative, cleaner signal).
- Update `docs/phase2_report.md` with a new §11 "Judge robustness and per-topic analysis" subsection and refresh the commentary in `reports/phase2_summary_commentary.md`.
- All new LLM calls use `--prompt cot` (the project default per `project_judge_cot_prompt_mode.md`); strict-mode is not used anywhere.

## Capabilities

### New Capabilities

- `per-topic-analysis`: per-topic F1 breakdown, cross-run topic-level diff, and a CLI surface for producing qualitative error inspections from existing run artefacts.

### Modified Capabilities

- `llm-judge`: adds a multi-judge rejudge contract — when more than one backend's records exist for the same `(qa_id, sentence_id, pmid)` triple, the expanded-qrels emitter SHALL be able to derive an intersection-pool variant alongside the union-pool default.
- `evaluation`: adds a third `--qrels-pool=intersection` value with semantics defined against the multi-judge records, and a topic-level aggregation knob alongside the existing per-cell macro.

## Impact

- New code: `src/trec_biogen/judge/intersection.py` (intersection-pool emitter), `src/trec_biogen/eval/per_topic.py` (topic aggregation), `scripts/per_topic_diff.py` (qualitative comparison CLI).
- Modified code: `src/trec_biogen/judge/rejudge.py` (multi-records persistence), `src/trec_biogen/eval/metrics.py` (intersection-pool path + topic agg), `src/trec_biogen/io/qrels.py` (parse multi-source records), `src/trec_biogen/eval/phase2_summary.py` (third pool column).
- New data: `data/qrels/biogen2025_taskA_qrels_expanded_together.jsonl` (sibling to the mini-cot file), `data/qrels/biogen2025_taskA_qrels_intersection.jsonl` (derived).
- New reports: `reports/judge_intersection_analysis.md`, `reports/per_topic_error_analysis.md`; new §11 in `docs/phase2_report.md`.
- LLM API spend: ~$0.50–1.50 (Together rate, ~5398 calls at ~$0.0001–0.0003 each).
- Compute: zero new GPU/CPU runs (re-scoring uses cached run artefacts).
- Tests: per-module unit tests under `tests/test_intersection_pool.py`, `tests/test_per_topic.py`.
- The §6.5 reproducibility anchor (`--qrels-pool=expanded --source=human` recovers the published 44.34) MUST continue to pass byte-for-byte — the intersection pool is strictly additive.
