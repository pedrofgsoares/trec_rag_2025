## 1. Engineering Cross-Cuts (lands first; unblocks every variant)

- [x] 1.1 Add `tqdm` progress bars to `src/trec_biogen/pipeline/phases.py` `segment_abstracts` inner loop (advance ≥ once per 1% of unique PMIDs)
- [x] 1.2 Add `tqdm` progress bars to `src/trec_biogen/nli/negation.py` `filter_negated` inner loop (advance ≥ once per 1% of sentences)
- [x] 1.3 Extend `src/trec_biogen/pipeline/metadata.py` to capture `wall_clock_seconds_per_phase` and `vram_peak_gb_per_phase` (reset `torch.cuda.max_memory_allocated` between phases)
- [x] 1.4 Extend `src/trec_biogen/pipeline/metadata.py` to record `phase2_variant`, `wall_clock_seconds_total`, `vram_peak_gb_total`, `judge_cost_usd` (zero by default) into `metadata.yaml`
- [x] 1.5 Add `--reuse-from=<run_dir>` CLI flag to `run_task_a.py` that symlinks intermediate Parquets from a prior run into the new run dir before `_maybe_run()` checks
- [x] 1.6 Update `tests/test_smoke_pipeline.py` to confirm tqdm and metadata fields are present without breaking the CI-safe smoke test

## 2. LLM-Judge Module (Vector 1 — methodological backbone)

- [ ] 2.1 Create package `src/trec_biogen/judge/` with `__init__.py`, `prompts.py`, `backends.py`, `validator.py`, `rejudge.py`
- [ ] 2.2 Design and implement the MedNLI-style prompt template in `prompts.py` (system + user + structured output schema); cap input abstract at 300 tokens
- [ ] 2.3 Implement `Backend` abstract base + `TogetherLlama70B` concrete (`meta-llama/Llama-3.1-70B-Instruct-Turbo`); honour `TOGETHER_API_KEY`
- [ ] 2.4 Implement `OpenAIMini` and `OpenAI4o` concrete backends; honour `OPENAI_API_KEY`
- [ ] 2.5 Implement `Judge.classify(answer_sentence, pmid, abstract_text)` returning the structured record `{label, confidence, input_tokens, output_tokens, backend}`
- [ ] 2.6 Skip-on-empty-abstract behaviour (deterministic `Not relevant` without backend call); record `skip_reason`
- [ ] 2.7 Implement concordance validator: classify the 588 human-labeled triples, compute per-class weighted F1, write `reports/llm_judge_validation.md` with confusion matrix
- [ ] 2.8 Implement concordance gate: exit non-zero if macro weighted F1 < 0.85 (configurable via `--threshold`)
- [ ] 2.9 Implement `rejudge.py` CLI: arguments `--submission` (which novel PMIDs to re-judge), `--backend`, `--cost-cap`, `--max-concurrent`
- [ ] 2.10 Implement cost accounting (per-call token + $ tracking) recorded into the rejudge run's `metadata.yaml`
- [ ] 2.11 Implement cost-cap abort path: graceful halt mid-run, partial expanded-qrels with `incomplete: true` flag
- [ ] 2.12 Implement expanded qrels emitter: copy human records unchanged, append LLM records with `source` and `confidence` fields; emit to `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`
- [ ] 2.13 Implement `compare-backends` subcommand on the 200-pair fixed sample
- [ ] 2.14 Unit tests: `tests/test_judge_prompts.py`, `tests/test_judge_validator.py` (against a recorded LLM-mock fixture), `tests/test_judge_expanded_qrels_shape.py`
- [ ] 2.15 Run concordance validation with the default Together backend; record the F1 numbers in `reports/llm_judge_validation.md`; if < 0.85, escalate to `OpenAIMini` and re-validate
- [ ] 2.16 Run the rejudge pass on the ~1100 novel PMIDs from `runs/20260516-134227-phase1_baseline/task_a_output.json`; produce `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`
- [ ] 2.17 Optional extension: rejudge BM25 top-30 per (qa_id, sentence_id) cell into an even broader expanded pool (bounded by `--cost-cap=$10`)

## 3. Evaluation: Dual-Pool Reporting (Vector 1 closure)

- [ ] 3.1 Add `--qrels-pool={official,expanded}` flag to `src/trec_biogen/eval/metrics.py` (default `official`)
- [ ] 3.2 Add `--source={human,llm,any}` filter to `eval/metrics.py` for restricting expanded qrels by source
- [ ] 3.3 Update `eval/qrels.py` to parse the optional `source` and `confidence` fields without breaking the official-pool path
- [ ] 3.4 Add `python -m trec_biogen.eval.phase2_summary` CLI that scans `runs/`, filters by `phase2_variant` in `metadata.yaml`, and writes `reports/phase2_summary.md` with the per-row schema (variant, official F1, expanded F1, Δ, wall-clock, VRAM, judge $)
- [ ] 3.5 Extend `eval/report.py` to write a row pairing `official` and `expanded` numbers when both are available
- [ ] 3.6 Re-score `runs/20260516-134227-phase1_baseline/task_a_output.json` on both pools and capture the Phase 2 starting line in the summary
- [ ] 3.7 Re-score the starter-kit run `runs/starter_baseline_20260514_150718/task_a_output.json` on both pools (sanity: the calibration anchor should be virtually unchanged on the expanded pool since baseline picks dominate it)
- [ ] 3.8 Update `tests/test_metrics.py` to cover both pools and the source filter

## 4. Variant 2a — `phase2_no_rerank`

- [ ] 4.1 Add `configs/run/phase2_no_rerank.yaml` inheriting `phase1_baseline` with `rerank: null` and `phase2_variant: no_rerank`
- [ ] 4.2 Make the rerank phase conditional in `run_task_a.py`: when `cfg.rerank is None`, pass `retrieval_support.parquet` directly to selection (no MedCPT-CE forward pass)
- [ ] 4.3 Run variant on the real input; collect timings; expect ≥ 30 pp lift on official-pool support F1
- [ ] 4.4 Append the row to `reports/phase2_summary.md`

## 5. Variant 2b — `phase2_no_negex`

- [ ] 5.1 Add `configs/run/phase2_no_negex.yaml` inheriting `phase1_baseline` with `nli.contradict.negex: false` and `phase2_variant: no_negex`
- [ ] 5.2 Make NegEx filtering optional in `run_task_a.py`: when `cfg.nli.contradict.negex is False`, pass `segmented_contradict.parquet` directly to the contradict NLI step
- [ ] 5.3 Run variant overnight (~10–12 h DeBERTa on ~2M pairs); resume mode reuses retrieval + segmentation parquets from a prior Phase 1 run
- [ ] 5.4 Append the row to `reports/phase2_summary.md`

## 6. Variant 2c — `phase2_allow_existing`

- [ ] 6.1 Add `configs/run/phase2_allow_existing.yaml` inheriting `phase1_baseline` with `selection.exclude_existing: false` and `phase2_variant: allow_existing`
- [ ] 6.2 Make existing-citation exclusion configurable in `src/trec_biogen/pipeline/selection.py` (currently hard-coded)
- [ ] 6.3 Run variant on the real input (cheap: < 1 h); validate that the official validator still passes (the track rule is enforced by the validator, not by our selection)
- [ ] 6.4 Append the row to `reports/phase2_summary.md`

## 7. Variant 3 — `phase2_scifive_large`

- [ ] 7.1 Add `configs/nli/scifive_large_contradict.yaml` pointing at `razent/SciFive-large-Pubmed_PMC-MedNLI` with `fp16: true`, `chunk_size: 4`
- [ ] 7.2 Pre-download SciFive-large via `curl -4` into `models/scifive-large-medNLI/` (avoid the IPv6/HF Hub stall from Phase 1)
- [ ] 7.3 Implement `score_contradict_pairs_t5(...)` in `src/trec_biogen/nli/stance.py` for the T5 seq2seq classification path (constrained-decoding over the three label tokens)
- [ ] 7.4 Add `configs/run/phase2_scifive_large.yaml` composing the contradict-NLI override + `phase2_variant: scifive_large`
- [ ] 7.5 Run variant overnight + day (~30 h); resume mode reuses retrieval, rerank, support NLI from a prior run
- [ ] 7.6 Append the row to `reports/phase2_summary.md`

## 8. Variant 4a — `phase2_bm25_rm3`

- [ ] 8.1 Add `configs/retrieval/bm25_rm3.yaml` enabling Pyserini RM3 query expansion (`fb_terms=10`, `fb_docs=10`, `original_query_weight=0.5` as defaults)
- [ ] 8.2 Extend `src/trec_biogen/retrieval/bm25.py` `BM25Index.search` to accept an optional `rm3=True` flag that sets the searcher's RM3 mode for that call
- [ ] 8.3 Add `configs/run/phase2_bm25_rm3.yaml` composing the retrieval override + `phase2_variant: bm25_rm3`
- [ ] 8.4 Run variant; collect timings; append the row to `reports/phase2_summary.md`

## 9. Variant 4b — `phase2_hybrid` (largest investment)

- [ ] 9.1 Write `scripts/select_top5m_pmids.py` that uses Pyserini's term-document frequency stats to pick the top-5M PMIDs by citation proxy; emits `data/interim/top5m_pmids.parquet`
- [ ] 9.2 Write `scripts/build_dense_index.sh` that encodes the selected 5M abstracts with `ncbi/MedCPT-Article-Encoder` (batch 8, CPU); emits `data/indexes/medcpt_5m/index.faiss` + `data/indexes/medcpt_5m/pmid_lookup.parquet`
- [ ] 9.3 Implement `src/trec_biogen/retrieval/dense.py` `DenseIndex` class with `.search(query_text, k)` returning ranked PMIDs
- [ ] 9.4 Implement `src/trec_biogen/retrieval/rrf.py` Reciprocal Rank Fusion with parameter `k=60` over two ranked lists
- [ ] 9.5 Add `configs/retrieval/hybrid_rrf.yaml` configuring the hybrid path
- [ ] 9.6 Make the orchestrator dispatch on retrieval flavour in `run_task_a.py`: BM25-only (default), BM25+RM3, or hybrid RRF
- [ ] 9.7 Add `configs/run/phase2_hybrid.yaml` composing the hybrid retrieval override + `phase2_variant: hybrid`
- [ ] 9.8 Run the one-off 5M encoding (~24 h CPU); verify FAISS index loads
- [ ] 9.9 Run `phase2_hybrid` variant; append the row to `reports/phase2_summary.md`
- [ ] 9.10 Unit tests: `tests/test_rrf.py`, `tests/test_dense_smoke.py` (env-gated like the BM25 round-trip test)

## 10. Consolidation and Reporting

- [ ] 10.1 Inspect `reports/phase2_summary.md`; identify the variant that maximises expanded-pool F1 on each class
- [ ] 10.2 Identify the variant whose Δ (official → expanded) is largest — this isolates the pool-bias contribution most clearly per architecture choice
- [ ] 10.3 Run the natural compositions of the top performers (e.g., `phase2_scifive_large` + `no_negex`) as a Phase 2.5 sanity check
- [ ] 10.4 Write `reports/phase2_summary.md` final commentary: which variants improved, by how much, on which pool, at what cost
- [ ] 10.5 Update `reports/llm_judge_validation.md` with final concordance numbers, including backend-comparison results if multiple backends were used
- [ ] 10.6 Update top-level `README.md` Phase 2 section: how to run variants, how to interpret dual-pool numbers, where the gap analysis lives
- [ ] 10.7 Verify the Phase 1 baseline gate still passes (`bash scripts/baseline_check.sh`) — Phase 2 must not regress reproducibility of §6.5

## 11. Phase 2 Sign-Off

- [ ] 11.1 Verify all six variants have a row in `reports/phase2_summary.md` with both pool numbers populated
- [ ] 11.2 Verify the LLM-judge concordance gate passed (`reports/llm_judge_validation.md` reports macro weighted F1 ≥ 0.85)
- [ ] 11.3 Verify at least one variant scores higher than the Phase 1 pipeline on the expanded pool (the success criterion that disproves the "everything was pool bias" null hypothesis)
- [ ] 11.4 Tag the repo `phase2-baseline` and archive the change via `/opsx:archive`
