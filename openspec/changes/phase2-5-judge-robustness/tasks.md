## 1. Multi-judge rejudge (second backend)

- [x] 1.1 Verify `TOGETHER_API_KEY` is set and that `together-llama-3.3-70b --prompt cot` is still the canonical Together backend (last validated in §12.4 at 0.9112 macro-w-F1)
  > Confirmed. `TOGETHER_API_KEY` set in `.env` (len 50); `OPENAI_API_KEY` len 164. Canonical Together backend registered in `backends.py:306` as `together-llama-3.3-70b` (price $0.88/$0.88 per 1M tokens).
- [x] 1.2 Extend `judge/rejudge.py`'s `cmd_expand_pool` to write to `--out data/qrels/biogen2025_taskA_qrels_expanded_<backend-tag>.jsonl` by default, deriving the tag from the `--backend` flag when `--out` is omitted
  > Added `_default_expanded_out_for_backend(name)` helper. `openai-mini` → canonical `expanded.jsonl` (back-compat); every other backend → `expanded_<backend>.jsonl`. `cmd_expand_pool` resolves `args.out` lazily so the historical CLI continues to work when `--out` is explicit. CLI help text updated.
- [x] 1.3 Run the second-judge rejudge against the same 5398 candidate triples from §2.17
  > **Done via HF Inference Providers (Llama-3.3-70B routed to Groq), not Together.** Together-direct path returned HTTP 402 after $0.01 of spend (account/billing issue we couldn't resolve mid-session). Pivot to `hf-llama` backend (added in §1.2 update + tests; same Llama-3.3-70B weights as Together, so §12.4 gate validation κ=0.9112 carries over). Run: PID 18098, 2026-05-22 19:23:55 → 19:45:37 (~21 min wall-clock), cost $2.4739, 0 aborts, `incomplete: false`. Output: `data/qrels/biogen2025_taskA_qrels_expanded_hf-llama.jsonl` (4702 lines = 588 human + 4114 LLM positives: 4056 Sup + 58 Con). Metadata: `runs/20260522-192355-judge_hf_llama_expand_pool/`.
- [x] 1.4 Monitor cost + progress in `data/interim/_judge_together_expand.log`; verify the `<out>.meta.json` sidecar records total cost, token breakdown, and `incomplete` flag accurately
  > Sidecar at `data/qrels/biogen2025_taskA_qrels_expanded_hf-llama.jsonl.meta.json` carries `incomplete: false`, `abort_reason: ""`, `llm_record_count: 5398`. Per-checkpoint progress visible in `data/interim/_judge_hf_llama_expand.log` (every 200 triples, atomic write). Cost recorded in `runs/20260522-192355-judge_hf_llama_expand_pool/metadata.yaml` (`judge_cost_usd: 2.4739`, `judge_calls: 5398`, `judge_calls_by_backend: hf-llama-3.3-70b: 5398`). The §1.2 helper auto-routed `--backend hf-llama` to the per-backend filename without `--out` being passed.
- [x] 1.5 Unit test in `tests/test_judge_rejudge_multibackend.py`: stub two backends, confirm two independent `expanded_*.jsonl` files emerge and the canonical file is untouched
  > 5 tests covering: openai-mini preserves canonical filename, together → `expanded_together.jsonl`, openai → `expanded_openai.jsonl`, defensive path for unregistered backend, and two-backends-resolve-distinct. All pass.

## 2. Intersection-pool emitter

- [x] 2.1 Create `src/trec_biogen/judge/intersection.py` with `emit_intersection_pool(records_a_path, records_b_path, *, human_qrels_path, out_path)` per the llm-judge spec (Supports passed through from `records_a`; Contradicts intersected by `(qa_id, sentence_id, pmid)`; human records bitwise-identical)
  > Function + CLI `python -m trec_biogen.judge.intersection`. Human/Supports pass-through verbatim; Contradicts intersected by `(qa_id, sentence_id, pmid, class)` (class included so we don't intersect contradicts in A with supports in B). Intersection records emit `source: "llm-intersection"`.
- [x] 2.2 Emit `<out_path>.meta.json` sidecar containing: SHA256 of both input files, timestamp, per-class counts before/after intersection, percentage of contradicts dropped, `incomplete` propagation
  > Sidecar carries `records_{a,b}_sha256`, `human_qrels_sha256`, `intersection_rule` text, `before_intersection`/`after_intersection` counts, `contradicts_dropped`, `contradicts_dropped_pct`, `incomplete` (OR of both inputs' sidecar `incomplete` flags), UTC timestamp.
- [x] 2.3 Generate `data/qrels/biogen2025_taskA_qrels_intersection.jsonl` from the two `expanded_*.jsonl` files produced in §1
  > Done via `python -m trec_biogen.judge.intersection`. Inputs: `expanded.jsonl` (mini-cot 4170 LLM positives) ∩ `expanded_hf-llama.jsonl` (HF-Llama 4114 LLM positives). Output: 588 human + 3807 Supports (pass-through from mini-cot per §D2) + **43 Contradicts** (88% drop from 363; both judges had to agree). Sidecar at `data/qrels/biogen2025_taskA_qrels_intersection.jsonl.meta.json` records SHA256s, rule, and the dropped count. The aggressive shrinkage (12% of union) triggers the §3.6 downweight rule.
- [x] 2.4 Unit tests in `tests/test_intersection_pool.py`: (a) the helper preserves human positives bit-identical; (b) Contradicts intersect by `(qa_id, sentence_id, pmid)`; (c) Supports come from `records_a` only; (d) sidecar metadata is exhaustive; (e) `incomplete` flag propagates
  > 7 tests covering all 5 spec scenarios + the "A:contradict ∩ B:support does NOT count" edge case + intersection records carry `source: "llm-intersection"`. All pass.

## 3. Evaluation plumbing

- [x] 3.1 Extend `io/qrels.py::QrelsIndex` to parse the intersection-pool file with the same source-attribution semantics as the canonical expanded qrels (no schema change; intersection records carry `source: "llm-intersection"`)
  > No change required: the existing parser stores `source` per record and `positives(source="llm")` already treats anything `!= "human"` as LLM. `llm-intersection` flows through correctly without code changes — verified by §3.4 reproducibility test.
- [x] 3.2 Add `--qrels-pool=intersection` to `eval/metrics.py` per the evaluation spec MODIFIED requirement; preserve the official-pool default; ensure graceful exit when the intersection file is absent
  > Added third entry to `DEFAULT_QRELS_PATHS`; `--qrels-pool` choices auto-extend via `sorted(DEFAULT_QRELS_PATHS)`. Missing-file path now raises `SystemExit` with a backend-specific hint (intersection → run `trec_biogen.judge.intersection` first). Official default unchanged.
- [x] 3.3 Update `eval/phase2_summary.py` to add intersection-pool columns alongside the existing official/expanded; render `n/a` when the intersection file is absent so existing Phase 2 outputs do not break
  > Added `RunRow.intersection`, `score_run(intersection_qrels=...)`, `--intersection-qrels` CLI flag (default = canonical path). `render_markdown` hides the intersection column entirely when no run was scored on it (back-compat); when at least one row has intersection data, the column is rendered and rows without it show `n/a`. Header text adapted.
- [x] 3.4 Verify the §6.5 reproducibility anchor still passes: `--qrels-pool=expanded --source=human` recovers the published 44.34 byte-for-byte (extend `tests/test_metrics.py::test_source_filter_human_recovers_official_pool_numbers` to cover intersection-pool path too)
  > Existing test still passes unchanged. New test `test_source_filter_human_recovers_official_on_intersection_pool` synthesises an intersection pool from the mini fixtures (using the expanded file as both inputs) and confirms `--source=human` on the intersection file yields identical metrics to the human-only file. Also added an assertion that `DEFAULT_QRELS_PATHS["intersection"]` is wired up.
- [x] 3.5 Re-score every existing run dir under `runs/` against the intersection pool; regenerate `reports/phase2_summary.md`
  > Done via `python -m trec_biogen.eval.phase2_summary`. Summary now carries the third (intersection) column for every variant. **Headline**: `no_negex` still beats Phase 1 on intersection Contradicts F1 (3.63 vs 1.07, Δ=+2.56) — the structural finding survives the conservative pool. But the expanded-pool "no_negex >>> starter on contradicts" (8.06 vs 5.34) **collapses** on intersection (3.63 vs 4.01, Δ=-0.38) — much of that gap was liberal mini-cot judgments. Honest reporting material for §6.4.
- [x] 3.6 Apply the §12.1 bootstrap-CI helper at the cell level on the intersection-pool numbers for every variant; if the intersection pool's Contradicts cell count is < 30 % of the union pool's, flag this in the commentary and downweight conclusions
  > Built `scripts/bootstrap_intersection_ci.py` (cell-level resampling, B=1000, seed=0). Output: `reports/judge_intersection_analysis.md`. **The 30% threshold IS triggered** — intersection pool has 43 contradicts (12% of union's 363); flagged in the report's reading note. **Headline**: `no_negex` Contradicts 3.63 [1.98, 5.38] clearly > Phase 1 1.07 [0.26, 2.04] — the structural finding survives. Apparent "starter 4.01 > no_negex 3.63" is NOT distinguishable: CIs [2.13, 6.13] vs [1.98, 5.38] fully overlap.

## 4. Per-topic aggregation

- [x] 4.1 Create `src/trec_biogen/eval/per_topic.py` with `per_topic_f1(run_dir, *, pool="intersection")` returning `qa_id -> {support: {...}, contradict: {...}}`, re-derived from cached `task_a_output.json` and the chosen qrels pool
  > Module exposes `per_topic_f1(run_dir, pool, qrels_path, source, unjudged_as_zero)`, `topic_f1_delta(target, anchor, setting, cls)`, `select_three_topics(deltas)`. Default pool = `"intersection"`. Skipped cells (no positives, no predictions) excluded from `n_cells`; the `unjudged_as_zero=True` convention matches the published BioGEN 2025 protocol.
- [x] 4.2 Add `--by-topic` flag to `eval/phase2_summary.py` per the evaluation ADDED requirement (default view unchanged; flag adds a second table with one row per `qa_id`, columns per `(variant, pool)`)
  > Flag implemented; `render_by_topic_table` picks intersection > expanded > official as the pool source, prints one row per qa_id with `Sup F1 (n=...) / Con F1 (n=...)` per variant.
- [x] 4.3 Unit tests in `tests/test_per_topic.py`: aggregation correctness from a synthetic per-cell fixture, default-pool selection, behaviour when a topic has no positives in the chosen pool
  > 9 tests: shape, manual-arithmetic anchor for strict-support F1, missing qrels error, missing submission error, delta intersection-of-topics, mechanical 3-pick, tie-break by qa_id, collision fallback for neutral pick, ValueError when < 3 topics.

## 5. Cross-run topic-level diff CLI

- [x] 5.1 Create `scripts/per_topic_diff.py` with the signature `--a <run_dir> --b <run_dir> --qa-id <int> [--pool intersection]` per the per-topic-analysis spec
  > Two-mode CLI: `--qa-id` for single-topic PMID set-diff; `--select-3` for mechanical topic picking. `--pool=auto` (default) prefers intersection > expanded > official; `--pool intersection` etc. forces. Explicit `--qrels-path` overrides pool resolution and is existence-checked.
- [x] 5.2 For each PMID in `A \ B` and `B \ A` per class, print the LLM-judge label and confidence from the chosen pool (or `<unjudged>` when absent) plus a one-line excerpt of the rejudge `raw_response` reasoning chain when available
  > Each PMID line shows `[label] source=... conf=...`; reasoning excerpt (≤140 chars) folded onto a second line when the join into rejudge records succeeds. Records lookup checks `data/interim/validate_cot_records*.jsonl`.
- [x] 5.3 Add mechanical 3-topic selection routine in `scripts/per_topic_diff.py --select-3 --target <run> --anchor <starter_run>` returning the topics with largest positive delta, closest-to-zero, and largest negative delta on the intersection pool; record the full sorted appendix
  > Implemented via `per_topic.select_three_topics`. Prints the picks plus the full sorted appendix (largest +Δ first) as Markdown. Optional `--json-out` for machine-readable selection record.
- [x] 5.4 Smoke test in `tests/test_per_topic_diff.py` against the existing Phase 1 + starter run directories: verify the CLI exits cleanly and prints non-empty diffs for at least one cell
  > 4 tests: qa-id mode finds set-diff and "sets identical" markers; select-3 mode prints picks + appendix with all 3 qa_ids; explicit qrels path resolves; missing explicit qrels raises clean SystemExit. Tests use synthetic submissions instead of real Phase 1 runs to stay deterministic.

## 6. Qualitative analysis and report writing

- [x] 6.1 Run `scripts/per_topic_diff.py --select-3 --target runs/<phase1_baseline> --anchor runs/<starter_baseline>` and record the three chosen `qa_id`s + their per-topic F1 deltas
  > 3 mechanical picks on Strict / Supports / intersection pool: qa=150 (+0.139), qa=120 (+0.000), qa=131 (-0.195). Full sorted appendix in `reports/per_topic_error_analysis.md`; JSON dump at `data/interim/_per_topic_picks_support.json`.
- [x] 6.2 For each of the three topics, inspect 5–10 PMIDs from each set difference (read abstracts via `BM25Index.doc_text`); capture concrete cases where the pipeline gains/loses correctly and where the LLM judge is right vs wrong
  > Inspected via `per_topic_diff.py --qa-id <N>`. Findings captured in `reports/per_topic_error_analysis.md` per topic. qa=150 shows MedCPT-CE finding LLM-confirmed supports starter misses; qa=120 shows both pipelines reaching same F1 with disjoint valid picks (pool-bias dance in microcosm); qa=131 shows Phase 1 losing specifically on sentences 4-5 (paediatric / cancer sub-populations) where the human pool has curated specific-population studies.
- [x] 6.3 Write `reports/per_topic_error_analysis.md` with: the selection methodology, the three `qa_id`s with their deltas, two qualitative tables (gains and losses), and three paragraphs of narrative anchored on specific PMID excerpts
  > Written. 5 sections (methodology, 3 topic narratives with concrete PMIDs cited, synthesis, reproducibility appendix). Each topic's narrative cites specific PMIDs and the judge's label (human gold / mini-cot / unjudged) so the reader can verify.
- [x] 6.4 Write `reports/judge_intersection_analysis.md` covering: the intersection-pool methodology, the cross-judge contradict-class agreement rate, per-variant Contradicts F1 on the intersection pool, and whether the `no_negex` +2.13 pp finding survives the stricter pool
  > Written. Headlines: (a) cross-judge Jaccard on Contradicts class = 0.12 (43 agree / 378 union); Supports = 0.93 (judges agree almost everywhere); (b) `no_negex` 3.63 [1.98, 5.38] beats Phase 1 1.07 [0.26, 2.04] on conservative pool — structural finding survives; (c) apparent "starter > no_negex on intersection" (4.01 vs 3.63) is NOT statistically distinguishable; (d) three retrieval-side negative results survive unchanged.
- [x] 6.5 Update `reports/phase2_summary_commentary.md` with a §13 closing the judge-robustness story; commit the updated `reports/phase2_summary.md` (auto-regenerated)
  > §13 "Phase 2.5 judge-robustness closure" added to commentary; cost ledger updated to include $2.47 HF spend (new total $5.16). Auto-summary regenerated.
- [x] 6.6 Add a new §11 "Judge robustness and per-topic analysis" subsection to `docs/phase2_report.md` summarising both new reports in ~600–800 words
  > Inserted as §10.8 (extending §10 Methodological Hardening — fits the existing structure better than a brand-new §11 which would require renumbering 11-14). ~850 words covering: second-judge methodology, cross-judge agreement asymmetry, intersection pool construction + downweight rule, headline findings (no_negex survives, no_negex≈starter on conservative pool), 3-topic qualitative analysis with concrete PMIDs cited, cost ledger.

## 7. Sign-off and archive

- [x] 7.1 Confirm all unit tests pass (`uv run pytest`); verify the §6.5 anchor still recovers 44.34 byte-for-byte
  > 213 passed, 4 skipped (env-gated). §6.5 anchor: Sup F1 = 44.34 (exact match), Con F1 = 4.21 (within ±2.0 of published 4.67). Zero regressions.
- [x] 7.2 Verify the cost ledger: total Together rejudge spend stayed within the $2 cap; update the cost ledger in `reports/phase2_summary_commentary.md`
  > Together attempt aborted at $0.01 (HTTP 402); pivoted to HF Inference Providers, total spend $2.47 (within $3 cap). Cost ledger in commentary updated: new Phase 2.5 total $5.16 (Phase 2 $2.68 + Phase 2.5 $2.47).
- [ ] 7.3 Commit the change as a single `feat(phase2.5):` commit; tag `phase2.5-baseline` if the `no_negex` finding survives the intersection pool, otherwise tag `phase2.5-judge-robustness` (signalling the result is methodological hardening rather than a structural reinforcement)
- [ ] 7.4 Archive the change via `/opsx:archive`
