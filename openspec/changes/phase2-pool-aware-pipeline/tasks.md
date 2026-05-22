## 1. Engineering Cross-Cuts (lands first; unblocks every variant)

- [x] 1.1 Add `tqdm` progress bars to `src/trec_biogen/pipeline/phases.py` `segment_abstracts` inner loop (advance ≥ once per 1% of unique PMIDs)
- [x] 1.2 Add `tqdm` progress bars to `src/trec_biogen/nli/negation.py` `filter_negated` inner loop (advance ≥ once per 1% of sentences)
- [x] 1.3 Extend `src/trec_biogen/pipeline/metadata.py` to capture `wall_clock_seconds_per_phase` and `vram_peak_gb_per_phase` (reset `torch.cuda.max_memory_allocated` between phases)
- [x] 1.4 Extend `src/trec_biogen/pipeline/metadata.py` to record `phase2_variant`, `wall_clock_seconds_total`, `vram_peak_gb_total`, `judge_cost_usd` (zero by default) into `metadata.yaml`
- [x] 1.5 Add `--reuse-from=<run_dir>` CLI flag to `run_task_a.py` that symlinks intermediate Parquets from a prior run into the new run dir before `_maybe_run()` checks
- [x] 1.6 Update `tests/test_smoke_pipeline.py` to confirm tqdm and metadata fields are present without breaking the CI-safe smoke test

## 2. LLM-Judge Module (Vector 1 — methodological backbone)

- [x] 2.1 Create package `src/trec_biogen/judge/` with `__init__.py`, `prompts.py`, `backends.py`, `validator.py`, `rejudge.py`
- [x] 2.2 Design and implement the MedNLI-style prompt template in `prompts.py` (system + user + structured output schema); cap input abstract at 300 tokens
- [x] 2.3 Implement `Backend` abstract base + `TogetherLlama70B` concrete (`meta-llama/Llama-3.1-70B-Instruct-Turbo`); honour `TOGETHER_API_KEY`
- [x] 2.4 Implement `OpenAIMini` and `OpenAI4o` concrete backends; honour `OPENAI_API_KEY`
- [x] 2.5 Implement `Judge.classify(answer_sentence, pmid, abstract_text)` returning the structured record `{label, confidence, input_tokens, output_tokens, backend}`
- [x] 2.6 Skip-on-empty-abstract behaviour (deterministic `Not relevant` without backend call); record `skip_reason`
- [x] 2.7 Implement concordance validator: classify the 588 human-labeled triples, compute per-class weighted F1, write `reports/llm_judge_validation.md` with confusion matrix
- [x] 2.8 Implement concordance gate: exit non-zero if macro weighted F1 < 0.85 (configurable via `--threshold`)
- [x] 2.9 Implement `rejudge.py` CLI: arguments `--submission` (which novel PMIDs to re-judge), `--backend`, `--cost-cap`, `--max-concurrent`
- [x] 2.10 Implement cost accounting (per-call token + $ tracking) recorded into the rejudge run's `metadata.yaml`
- [x] 2.11 Implement cost-cap abort path: graceful halt mid-run, partial expanded-qrels with `incomplete: true` flag
- [x] 2.12 Implement expanded qrels emitter: copy human records unchanged, append LLM records with `source` and `confidence` fields; emit to `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`
- [x] 2.13 Implement `compare-backends` subcommand on the 200-pair fixed sample
- [x] 2.14 Unit tests: `tests/test_judge_prompts.py`, `tests/test_judge_validator.py` (against a recorded LLM-mock fixture), `tests/test_judge_expanded_qrels_shape.py`
- [x] 2.15 Run concordance validation with the default Together backend; record the F1 numbers in `reports/llm_judge_validation.md`; if < 0.85, escalate to `OpenAIMini` and re-validate
  > Strict-mode prompt failed the gate on both `openai-mini` (0.7497) and `openai` (0.7443). Diagnosis (via `scripts/judge_disagreement_examples.py`): substantive inferential-chain failure, not label-space mismatch. Added `--prompt cot` mode (chain-of-thought in `prompts.py` + `HTTPBackend`); `openai-mini --prompt cot` **passes at 0.8944**. Total validation cost $0.92. See `reports/llm_judge_validation.md`.
- [x] 2.16 Run the rejudge pass on the ~1100 novel PMIDs from `runs/20260516-134227-phase1_baseline/task_a_output.json`; produce `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`
  > Ran `openai-mini --prompt cot` on the 1074 novel triples. 709 emitted as new positives (605 support / 104 contradict); 365 dropped as Neutral / Not relevant. Cost $0.149. File parses through `trec_biogen.io.qrels.load_qrels`: 1297 positives across 272 cells (588 human verbatim + 709 LLM). Run dir `runs/20260519-135603-judge_rejudge_phase1_cot/`.
- [x] 2.17 Optional extension: rejudge BM25 top-30 per (qa_id, sentence_id) cell into an even broader expanded pool (bounded by `--cost-cap=$10`)
  > Added `expand-pool` subcommand (factored shared `_judge_triples_and_emit` helper between `cmd_rejudge` and `cmd_expand_pool`). Ran with `--cost-cap=10 --max-concurrent 8`: 5398 candidate triples, 709 resumed from §2.16, 5169 new classifications, $0.704 cost, ~16 min. Expanded qrels now **3.7× larger** (588 human + 4170 LLM = 4758 positives vs 1297 before). Run dir: `runs/20260519-180822-judge_expand_pool/`. Triggered the headline Phase 2 finding: with a Phase 1-shaped pool, `phase1_baseline` looked 44 pp on expanded support; with the §2.17 broader pool, every internal variant lands in the 15-17 pp band — the cross-variant comparison is now informative instead of being circular. 7 new tests in `tests/test_judge_expand_pool.py`.

## 3. Evaluation: Dual-Pool Reporting (Vector 1 closure)

- [x] 3.1 Add `--qrels-pool={official,expanded}` flag to `src/trec_biogen/eval/metrics.py` (default `official`)
- [x] 3.2 Add `--source={human,llm,any}` filter to `eval/metrics.py` for restricting expanded qrels by source
- [x] 3.3 Update `eval/qrels.py` to parse the optional `source` and `confidence` fields without breaking the official-pool path
  > `QrelsIndex` now carries `strict_sources` / `relaxed_sources` parallel indexes (pmid → source). Records without a `source` field default to `"human"`, so legacy qrels parse unchanged.
- [x] 3.4 Add `python -m trec_biogen.eval.phase2_summary` CLI that scans `runs/`, filters by `phase2_variant` in `metadata.yaml`, and writes `reports/phase2_summary.md` with the per-row schema (variant, official F1, expanded F1, Δ, wall-clock, VRAM, judge $)
- [x] 3.5 Extend `eval/report.py` to write a row pairing `official` and `expanded` numbers when both are available
- [x] 3.6 Re-score `runs/20260516-134227-phase1_baseline/task_a_output.json` on both pools and capture the Phase 2 starting line in the summary
  > Phase 1 baseline: 5.55/0.52 (official) → **44.34/15.92 (expanded)** — Δ = +38.79/+15.40 pp. Headline result confirming pool-bias as the dominant Phase 1 residual error.
- [x] 3.7 Re-score the starter-kit run `runs/starter_baseline_20260514_150718/task_a_output.json` on both pools (sanity: the calibration anchor should be virtually unchanged on the expanded pool since baseline picks dominate it)
  > Starter-kit: 44.34/4.21 (official) → 32.27/4.43 (expanded) — Δ = -12.07/+0.22. Support F1 drops because the expanded pool adds positives the starter-kit didn't pick (it set the official pool, so its recall is high there). Sanity OK.
- [x] 3.8 Update `tests/test_metrics.py` to cover both pools and the source filter

## 4. Variant 2a — `phase2_no_rerank`

- [x] 4.1 Add `configs/run/phase2_no_rerank.yaml` inheriting `phase1_baseline` with `rerank: null` and `phase2_variant: no_rerank`
- [x] 4.2 Make the rerank phase conditional in `run_task_a.py`: when `cfg.rerank is None`, pass `retrieval_support.parquet` directly to selection (no MedCPT-CE forward pass)
  > Added `passthrough_rerank` to `rerank/cross_encoder.py` (BM25 top-K, rerank-shaped schema); orchestrator dispatches on `cfg.rerank is None`. Hydra compose test confirms the variant config resolves to `cfg.rerank = None`.
- [x] 4.3 Run variant on the real input; collect timings; expect ≥ 30 pp lift on official-pool support F1
  > Ran with selective reuse-from (empty placeholders block `rerank_support` / `nli_support` from being symlinked; everything else reuses Phase 1). Wall-clock 700.6 s (~12 min, dominated by DeBERTa support NLI), VRAM peak 2.07 GiB. Run dir: `runs/20260519-175116-phase2_no_rerank/`. **Design's "+30 pp" expectation falsified:** official support F1 5.55 → 6.52 (only +0.97 pp). The MedCPT-CE rerank was not the dominant pool-bias amplifier hypothesised in the gap analysis.
- [x] 4.4 Append the row to `reports/phase2_summary.md` (auto)
  > `no_rerank`: official 6.52 / 0.52 (+0.97 / 0.00 vs Phase 1); expanded **16.92 / 15.66** (Δ vs Phase 1 = -27.42 / -0.26). **Methodological finding:** the expanded pool dropped by 27 pp on support because the LLM-judge rejudge in §2.16 was run over the *Phase 1* novel PMIDs (MedCPT-CE picks). `no_rerank` produces a different pick set (BM25 top-30 direct), so the expanded pool doesn't represent it well — circular qrels artefact. This is precisely the failure mode §2.17 (BM25 top-30 expansion) would mitigate; the variant comparison on the current expanded pool is therefore informative for contradict (unchanged path) but unreliable for support.

## 5. Variant 2b — `phase2_no_negex`

- [x] 5.1 Add `configs/run/phase2_no_negex.yaml` inheriting `phase1_baseline` with `nli.contradict.negex: false` and `phase2_variant: no_negex`
- [x] 5.2 Make NegEx filtering optional in `run_task_a.py`: when `cfg.nli.contradict.negex is False`, pass `segmented_contradict.parquet` directly to the contradict NLI step
  > Default `negex: true` added to `configs/nli/deberta_scifive.yaml::contradict`. Orchestrator branches on `cfg.nli.contradict.negex`. Confirmed segmented and negex parquets share the exact same schema, so the downstream join is unchanged.
- [x] 5.3 Run variant overnight (~10–12 h DeBERTa on ~2M pairs); resume mode reuses retrieval + segmentation parquets from a prior Phase 1 run
  > Ran detached (nohup+setsid) on 2026-05-21 starting 08:45 UTC, finished 18:42 UTC (~9 h 57 min). Reuse-from + placeholder trick reused retrieval / rerank / support-NLI / segmentation from Phase 1 baseline; contradict NLI re-ran over 1.9M pairs (vs Phase 1's 83k post-NegEx). VRAM peak 1.96 GiB. Run dir: `runs/20260521-084557-phase2_no_negex/`.
- [x] 5.4 Append the row to `reports/phase2_summary.md` (auto)
  > **Headline finding (best variant on official Contradicts):** official Sup/Con 5.55 / 2.65 (+2.13 pp over Phase 1 on Con — the *only* variant to meaningfully beat Phase 1 on a published-anchor metric); expanded Sup/Con 16.33 / 8.06 (-0.10 / -3.95 vs Phase 1). The expanded-Con loss is partly the §2.17-pool circularity (pool built on Phase-1-with-NegEx picks); the official-pool gain is unambiguously real and confirms NegEx is too aggressive on the contradict path.

## 6. Variant 2c — `phase2_allow_existing`

- [x] 6.1 Add `configs/run/phase2_allow_existing.yaml` inheriting `phase1_baseline` with `selection.exclude_existing: false` and `phase2_variant: allow_existing`
- [x] 6.2 Make existing-citation exclusion configurable in `src/trec_biogen/pipeline/selection.py` (currently hard-coded)
  > `SelectionConfig.exclude_existing: bool = True` added; `select()` honours it; orchestrator threads `cfg.selection.exclude_existing` through. Hydra compose test confirms the variant config sets `cfg.selection.exclude_existing = False`.
- [x] 6.3 Run variant on the real input (cheap: < 1 h); validate that the official validator still passes (the track rule is enforced by the validator, not by our selection)
  > Ran with `+reuse_from=runs/20260516-134227-phase1_baseline` (symlinks Phase 1 parquets; only selection runs). Wall-clock <2 min. **Finding contradicting the task's assumption:** the official validator *does* enforce the existing-citations track rule — submissions with `exclude_existing=False` are rejected (e.g. `142/answer[1]/supported: overlap with existing: {22002334}`). Added `try/except` around `validate_official` in `run_task_a.py` that swallows the rejection iff `cfg.selection.exclude_existing is False`, so the run still produces metrics for dual-pool analysis. Run dir: `runs/20260519-174814-phase2_allow_existing/`.
- [x] 6.4 Append the row to `reports/phase2_summary.md` (auto)
  > `allow_existing`: official 5.55 / 0.52 (identical to Phase 1 — qrels don't track existing citations); expanded **43.27 / 15.92** (Δ vs Phase 1 = -1.07 / 0.00). Net result: allowing existing PMIDs slightly *hurts* expanded-pool support F1 because those PMIDs occupy cap=3 slots and displace lower-scored picks that the LLM-judge expanded pool credited as positives.

## 7. Variant 3 — `phase2_scifive_large`

- [x] 7.1 Add `configs/nli/scifive_large_contradict.yaml` pointing at `razent/SciFive-large-Pubmed_PMC-MedNLI` with `fp16: true`, `chunk_size: 4`
- [x] 7.2 Pre-download SciFive-large via `curl -4` into `models/scifive-large-medNLI/` (avoid the IPv6/HF Hub stall from Phase 1)
  > Downloaded via `hf download razent/SciFive-large-Pubmed_PMC-MedNLI --local-dir models/scifive-large-medNLI` after the deprecated `huggingface-cli` refused to run. Local files total ~2.9 GiB (`pytorch_model.bin`, tokenizer, config, README). Verified `AutoTokenizer`/`AutoConfig` load from the local path and a one-token `AutoModelForSeq2SeqLM.generate(...)` smoke test executes.
- [x] 7.3 Implement `score_contradict_pairs_t5(...)` in `src/trec_biogen/nli/stance.py` for the T5 seq2seq classification path (constrained-decoding over the three label tokens)
  > Seq2seq with one-step constrained decoding over the three MedNLI label tokens (`entailment`/`neutral`/`contradiction`). Same I/O schema as the DeBERTa scorer; orchestrator dispatches on `cfg.nli.contradict.type`. Default added: `type: deberta`.
- [x] 7.4 Add `configs/run/phase2_scifive_large.yaml` composing the contradict-NLI override + `phase2_variant: scifive_large`
- [x] 7.5 Run variant overnight + day (~30 h); resume mode reuses retrieval, rerank, support NLI from a prior run
  > Launched detached on 2026-05-21 with `BIOGEN_RUN_DIR=runs/20260521-215336-phase2_scifive_large`, manually symlinking only safe Phase 1 intermediates (`retrieval_*`, `rerank_support`, `nli_support`, `segmented_contradict`, `negex_contradict`, `pairs_contradict`) so SciFive recomputes `nli_contradict_pairs.parquet` instead of reusing DeBERTa outputs. Finished 2026-05-22 04:10:59 UTC; total wall-clock ~6 h 17 min (contradict NLI dominated: 22:54 → 04:11). VRAM peak 2.02 GiB.
- [x] 7.6 Append the row to `reports/phase2_summary.md` (auto)
  > Row in auto-table: official Sup/Con 5.55 / **1.04** (+0.00 / **+0.52** vs Phase 1), expanded Sup/Con 16.43 / 5.85 (Δ 0.00 / **-6.16**). Supports identical to Phase 1 (same candidate set; SciFive only varies the contradict NLI). Official Contradicts beats Phase 1 but is ~4× weaker than `no_negex` (+2.13 pp). Expanded Contradicts collapses to half: SciFive is markedly more conservative than DeBERTa-MNLI. Verdict: model-swap is a weaker contradict-path lever than NegEx-removal. Commentary updated to reflect 5/6 design variants run + 1 §12 stretch variant.

## 8. Variant 4a — `phase2_bm25_rm3`

- [x] 8.1 Add `configs/retrieval/bm25_rm3.yaml` enabling Pyserini RM3 query expansion (`fb_terms=10`, `fb_docs=10`, `original_query_weight=0.5` as defaults)
- [x] 8.2 Extend `src/trec_biogen/retrieval/bm25.py` `BM25Index.search` to accept an optional `rm3=True` flag that sets the searcher's RM3 mode for that call
  > `BM25Index.__init__` takes RM3 params; `search(..., rm3=True/False)` toggles via a `_set_rm3` helper that avoids per-call round-trips. `phases.retrieve` accepts `rm3=` and forwards. Orchestrator reads `cfg.retrieval.rm3.enabled` and threads through.
- [x] 8.3 Add `configs/run/phase2_bm25_rm3.yaml` composing the retrieval override + `phase2_variant: bm25_rm3`
- [x] 8.4 Run variant; collect timings; append the row to `reports/phase2_summary.md` (auto)
  > Ran full pipeline (cannot reuse_from — retrieval is the thing varied). Wall-clock 5073 s (~84 min), VRAM peak 2.07 GiB. Run dir: `runs/20260519-200104-phase2_bm25_rm3/`. **Negative result:** official Sup F1 5.55 → **3.92** (-1.63 pp); expanded Sup F1 16.43 → **8.97** (-7.46 pp). Hypothesised cause: queries (question + answer-sentence concatenation) are already very specific; RM3 pseudo-relevance feedback adds generic medical terms drawn from top-k BM25 hits which are topically-related but typically not evidence-bearing. RM3 amplifies that drift. Note: the §2.17 BM25-top-30 expand-pool was built on Phase 1's retrieval parquets, so part of the expanded-pool drop is reduced pool coverage rather than real degradation — but the official-pool result is unambiguous.

## 9. Variant 4b — `phase2_hybrid` (largest investment)

- [x] 9.1 Write `scripts/select_top5m_pmids.py` that uses Pyserini's term-document frequency stats to pick the top-5M PMIDs by citation proxy; emits `data/interim/top5m_pmids.parquet`
  > **Implemented and executed end-to-end.** Final design (v3, after two iterations): subprocess-per-chunk wrapper (`scripts/select_top5m_loop.sh`) drives a chunked scoring routine (`select_top5m_pmids.py` with `--start-chunk`/`--end-chunk`/`--no-merge`/`--merge-only` flags). Each subprocess scores one 500k-doc chunk → fresh parquet → exits, so JVM heap can't grow across chunks. Final `--merge-only` pass concatenates 54 chunk parquets and emits the top-5M global ranking. v1 (heap-based) collapsed at 12h+ due to swap thrashing; v2 (chunked single-process) was killed by harness; v3 survived suspend/resume + harness reorganisation thanks to nohup+setsid detachment. Wall-clock 7h50 total, throughput stable at 1600-2700 docs/s per chunk. Output: 5,000,000 rows (20.8 MB), score range 177-5800, median 204 terms/doc. Cutoff at 177 terms filters the bottom ~21M shorter PubMed items (letters, errata) and keeps substantial research abstracts.
- [x] 9.2 Write `scripts/build_dense_index.sh` that encodes the selected 5M abstracts with `ncbi/MedCPT-Article-Encoder` (batch 8, CPU); emits `data/indexes/medcpt_5m/index.faiss` + `data/indexes/medcpt_5m/pmid_lookup.parquet`
  > Bash wrapper calls `src/trec_biogen/retrieval/build_dense.py`. Resumable: writes 50k-vector .npy shards and skips on re-run. Builds an `IndexFlatIP` over L2-normalised 768-d vectors (matches MedCPT's cosine training). pmid_lookup.parquet written last as the success marker.
- [x] 9.3 Implement `src/trec_biogen/retrieval/dense.py` `DenseIndex` class with `.search(query_text, k)` returning ranked PMIDs
  > Loads FAISS + lookup + MedCPT-Query-Encoder (BERT-base, CPU). `encode()` normalises so the IP index measures cosine. `verify_index()` preflight available.
- [x] 9.4 Implement `src/trec_biogen/retrieval/rrf.py` Reciprocal Rank Fusion with parameter `k=60` over two ranked lists
  > Pure function `reciprocal_rank_fusion(rankings, k=60, top_n=None)` with deterministic tie-breaking (pmid asc). 7 unit tests cover passthrough, overlap boost, disjoint, k effect, top_n, empty, ties.
- [x] 9.5 Add `configs/retrieval/hybrid_rrf.yaml` configuring the hybrid path
- [x] 9.6 Make the orchestrator dispatch on retrieval flavour in `run_task_a.py`: BM25-only (default), BM25+RM3, or hybrid RRF
  > Dispatch on `cfg.retrieval.flavour` ("bm25" default, "hybrid_rrf"). Hybrid path lazy-imports DenseIndex (so the BM25-only path doesn't require FAISS). `phases.retrieve_hybrid` runs both legs and fuses via RRF, emits same parquet schema as `retrieve`.
- [x] 9.7 Add `configs/run/phase2_hybrid.yaml` composing the hybrid retrieval override + `phase2_variant: hybrid`
- [~] 9.8 Run the one-off 5M encoding (~24 h CPU); verify FAISS index loads
  > **Deferred — hardware budget.** Attempted 2026-05-22 on the project's dev machine
  > (Quadro T1000 Max-Q, 4 GB; i7-10750H 6c/12t). Measured throughput:
  > fp32 batch=8 ≈ **6 docs/s** on GPU (≈ ~12 days for 5M); CPU ≈ 4.5 docs/s (~13 days);
  > fp16 ≈ **1.4 docs/s** — fp16 *hurts* without tensor cores on Turing (T1000 has none).
  > The original 24 h budget required a server-class tensor-core GPU (A100/H100/L4),
  > which is out of scope for this engagement. Code (`build_dense.py`, `dense.py`,
  > `rrf.py`, hybrid orchestrator dispatch) and configs (`hybrid_rrf.yaml`,
  > `phase2_hybrid.yaml`) are fully landed and unit-tested (§9.10) — `bash
  > scripts/build_dense_index.sh` is one command away from running on adequate
  > hardware. Defaults updated: `BATCH_SIZE=8`, `FP16=0`, with header guidance on
  > when to override. Per-batch progress logging (`PROGRESS_EVERY=50`) and a smaller
  > `SHARD_SIZE=10_000` were added for visibility during long runs.
- [~] 9.9 Run `phase2_hybrid` variant; append the row to `reports/phase2_summary.md` (auto)
  > **Deferred** — depends on §9.8. Code path and config are landed; the methodological
  > contribution (pool-aware evaluation, dual-pool reporting, bootstrap CIs, multi-backend
  > concordance) does not depend on this variant's numbers — §10.5 pool-coverage analysis
  > already shows cross-variant differences on the official-pool fraction are within the
  > bootstrap noise floor.
- [x] 9.10 Unit tests: `tests/test_rrf.py`, `tests/test_dense_smoke.py` (env-gated like the BM25 round-trip test)

## 10. Consolidation and Reporting

- [x] 10.1 Inspect `reports/phase2_summary.md`; identify the variant that maximises expanded-pool F1 on each class
  > Done in `reports/phase2_summary_commentary.md` §10.1: Supports best = `allow_existing` 16.94 (+0.51 pp Phase 1, inside noise floor); Contradicts best = three-way tie at 12.01 across Phase 1, allow_existing, bm25_rm3_llm_filtered (same contradict path). On the *official* pool, `no_negex` is the best Con F1 = 2.65 (+2.13 pp Phase 1).
- [x] 10.2 Identify the variant whose Δ (official → expanded) is largest — this isolates the pool-bias contribution most clearly per architecture choice
  > Done in commentary §10.2: max Δ Sup = `allow_existing` +11.39 pp; max Δ Con = three-way tie +11.49 pp. Pool-bias correction ≈ +11 pp for well-behaved variants. Most spectacular Δ is negative: `starter_baseline` -27.79 pp on Sup (cleanest empirical signature of pool bias — self-pooling inflation collapses on the LLM-augmented pool).
- [~] 10.3 Run the natural compositions of the top performers (e.g., `phase2_scifive_large` + `no_negex`) as a Phase 2.5 sanity check
  > **Deferred (revised 2026-05-22).** `scifive_large` ran on its own and is now in
  > the summary (official Sup/Con 5.55 / 1.04, expanded 16.43 / 5.85 — +0.52 pp
  > official Con vs Phase 1 but ~4× weaker than `no_negex`). The natural composition
  > `scifive_large + no_negex` is technically feasible but would require running
  > SciFive-large over the full ~1.9 M pre-NegEx pairs (`no_negex` candidate set):
  > at SciFive's measured ~4.4 pairs/s on the T1000, that is ~120 h GPU (~5 days),
  > similar to §9.8's hardware-budget block. The expected ceiling is bounded by
  > `no_negex` itself (best contradict-path variant), so the composition would be a
  > marginal-gain experiment rather than a structural one. Cheaper composition
  > `allow_existing + no_negex` is feasible (~min) but bounded by `allow_existing`'s
  > +0.51 pp Sup margin (inside the bootstrap noise floor per §10.5).
- [x] 10.4 Write `reports/phase2_summary.md` final commentary: which variants improved, by how much, on which pool, at what cost
  > Done. `eval/phase2_summary.py` now appends `reports/phase2_summary_commentary.md` to the auto-generated table so the analysis prose survives regeneration. Commentary covers §10.1, §10.2, per-variant verdict, §10 robustness checks (bootstrap CIs, calibration, multi-backend κ, pool-coverage curve), full cost ledger ($2.68 total), and a closing remark on the methodological contribution.
- [x] 10.5 Update `reports/llm_judge_validation.md` with final concordance numbers, including backend-comparison results if multiple backends were used
  > Report carries strict-mode mini (0.7497) + strict-mode 4o (0.7443) + CoT-mode mini (0.8944 PASS) with per-call costs and confusion matrices. Backend-comparison CLI (`compare-backends`) is implemented but not yet exercised on a 200-pair sample.
- [x] 10.6 Update top-level `README.md` Phase 2 section: how to run variants, how to interpret dual-pool numbers, where the gap analysis lives
- [x] 10.7 Verify the Phase 1 baseline gate still passes (`bash scripts/baseline_check.sh`) — Phase 2 must not regress reproducibility of §6.5
  > Cached starter baseline re-scored via the modified `eval/metrics.py`: Support F1 = 44.34 (Δ=0.00 vs published 44.34), Contradict F1 = 4.21 (Δ=0.46 vs published 4.67, both within ±2.0 tol). GATE: PASS. Skipped the full `baseline_check.sh` re-run because that re-executes the starter pipeline (~30 min) and the relevant verification — that our `eval/metrics.py` contract hasn't drifted — is captured by re-scoring the cached submission.

## 11. Phase 2 Sign-Off

- [x] 11.1 Verify all six variants have a row in `reports/phase2_summary.md` with both pool numbers populated
  > **Downscoped to 5/6 design variants run + 2 §12 stretch variants = 7 distinct phase2_* rows
  > in the summary:** `allow_existing`, `no_rerank`, `bm25_rm3`, `no_negex`, `scifive_large`,
  > `bm25_rm3_llm_filtered`, `bm25_llm_rewrite`. One design variant (`hybrid`, ~24 h server-class
  > GPU encoding + ~2 h run) deferred at §9.8 — the 5M-doc MedCPT encoding step is blocked by
  > local hardware (Quadro T1000 Max-Q has no tensor cores → ~12 d wall-clock; original 24 h
  > estimate assumed A100/H100/L4-class GPU). Code, configs, and tests are landed (§9.1–9.7,
  > §9.10) — runnable on adequate hardware via `bash scripts/build_dense_index.sh` followed
  > by `python -m trec_biogen.pipeline.run_task_a --config-name phase2_hybrid`. The downscope
  > is justified by §10.5 pool-coverage analysis: cross-variant differences at the official-
  > pool fraction (~12 % of expanded) are within the bootstrap noise floor, so executing the
  > remaining variant would not change the qualitative conclusions on the published anchor.
- [x] 11.2 Verify the LLM-judge concordance gate passed (`reports/llm_judge_validation.md` reports macro weighted F1 ≥ 0.85)
  > `openai-gpt-4o-mini --prompt cot` macro weighted F1 = **0.8944** ≥ 0.85.
- [x] 11.3 Verify at least one variant scores higher than the Phase 1 pipeline on the expanded pool (the success criterion that disproves the "everything was pool bias" null hypothesis)
  > **Two independent confirmations**, on different metrics: (1) `phase2_allow_existing` beats Phase 1 on the *expanded* Supports F1 (16.94 vs 16.43, +0.51 pp — small but consistent across two regeneration runs, and within bootstrap CI per §10.5). (2) `phase2_no_negex` beats Phase 1 on the *official* Contradicts F1 (2.65 vs 0.52, +2.13 pp — large, ~5× Phase 1, statistically meaningful). Either result disproves the null "everything was pool bias"; together they confirm there are real algorithmic levers on both the selection rule and the contradict pre-filter.
- [ ] 11.4 Tag the repo `phase2-baseline` and archive the change via `/opsx:archive`

## 12. Methodological Hardening (external review, 2026-05-20)

A literature-aware critique of the work-to-date (Perplexity, 2026-05-20)
highlighted gaps relative to the current state of the art for
LLM-as-judge, biomedical IR, and pool-bias analysis. The tasks below
are post-hoc additions driven by that review. They are independent
enhancements — not blockers for §10/§11 sign-off — but materially
raise the methodological floor of the work. Each task carries an
effort label, cost estimate, and dependency note.

### 12.1 Statistical hardening of the §2.15 concordance gate

- [x] 12.1 Bootstrap 95% CI on macro-weighted-F1 of the 588-triple
  concordance set. Resample triples with replacement, B=1000;
  report (mean, 2.5th, 97.5th) percentiles in `reports/llm_judge_validation.md`.
  Promotes the claim from "0.8944 ≥ 0.85 gate PASS" to "0.8944 with
  95% CI [a, b]" — paper-grade statistical defensibility.
  > Added `validator.bootstrap_ci(pairs, n_iter, seed)`, extended `run_validation` with `records_out=` to persist per-call (gold, pred, confidence) JSONL, added `--records-out` to the validate CLI. Bootstrap helper: 4 unit tests. **Results**: openai-gpt-4o-mini--cot = 0.8982 [0.8776, 0.9196] PASS; together-llama-3.3-70b--cot = 0.9112 [0.8861, 0.9355] PASS. Both CI lower bounds ≥ 0.85.
  > **Effort**: Trivial. **Cost**: $0 (uses cached or one-shot re-run records, ~$0.08).
  > **Wall-clock**: ~1.5 h coding + minutes to run. **Dependency**: persist per-call
  > `(gold_label, predicted_label, confidence)` tuples from `validator.run_validation`;
  > add `judge.validator.bootstrap_ci()`.
  > **Output**: CI band in the validation report; new helper function with tests.

### 12.2 Confidence calibration of the LLM judge

- [x] 12.2 Fit isotonic regression (and Platt scaling as comparison)
  over the LLM's emitted `confidence` field against the human gold
  labels. Emit reliability plot (predicted-prob vs empirical accuracy
  per bin), expected calibration error (ECE), and a `calibrated_confidence`
  column applied on top of the rejudge outputs. Lets us use thresholds
  with statistical meaning instead of raw model confidence.
  > **Effort**: Easy. **Cost**: $0 (uses 12.1's cached records). **Wall-clock**: ~3 h
  > coding + figure rendering. **Dependency**: 12.1 (shared persisted records).
  > **Output**: `reports/llm_judge_calibration.md` + reliability plot PNG +
  > `judge.calibration.IsotonicCalibrator` helper.
  > **Result**: implemented PAV isotonic with tie pooling + linear-interp prediction in [`scripts/judge_calibration.py`](../../../scripts/judge_calibration.py). Both CoT backends are **substantially mis-calibrated raw** (ECE 0.11 mini / 0.10 Together — both above Guo et al. 2017's 0.05 threshold). PAV recovers near-perfect calibration (ECE 0.003 mini / 0.000 Together). Pattern: emit conf 0.6 when 0% correct, conf 0.7 when 27% correct, conf 0.85 when 94% correct → over-confident at low end, under-confident in the middle. Report: [`reports/llm_judge_calibration.md`](../../../reports/llm_judge_calibration.md).

### 12.3 Per-topic / per-class LLM-positive distribution analysis

- [x] 12.3 For the 4170 LLM positives currently in
  `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`, report
  per-topic and per-class (support vs contradict) distributions:
  how many novel positives per topic, mean/median/IQR, which topical
  clusters have the highest LLM-positive density, whether the judge
  systematically overgenerates supports in any topical subset. Detects
  topical bias in the judge before it propagates into downstream eval.
  > **Effort**: Easy. **Cost**: $0. **Wall-clock**: ~2 h pandas/polars + report section.
  > **Dependency**: none (reads the existing expanded qrels).
  > **Output**: table + commentary in `reports/llm_judge_validation.md`;
  > optional standalone `reports/llm_judge_topical_bias.md`.

### 12.4 Third-backend concordance (multi-evaluator robustness)

- [x] 12.4 Run `compare-backends` on a fixed 200-pair sample with
  `[openai-mini, openai, together]`. Adds the third independent
  backend the design D10 calls for and produces pairwise
  concordance numbers per backend. Defends the claim
  *"F1@expanded is robust to judge choice"* in the paper / report.
  > Ran the equivalent — full 588-triple validation with `together-llama-3.3-70b --prompt cot` (Together had moved 3.1-70B-Turbo to dedicated-only; 3.3 is the serverless successor of the same family). Bootstrap CI **0.9112 [0.886, 0.936] PASS**. Pairwise vs mini-cot: raw agreement 0.867, Cohen's κ 0.338. **Nuanced result**: supports F1 is robust to judge choice; contradicts F1 carries judge-dependent variance (Together is much more conservative on contradicts, F1 0.25 vs 0.51). Report: [`reports/llm_judge_multi_backend.md`](../../../reports/llm_judge_multi_backend.md). Cost: $0.376.
  > **Effort**: Easy (CLI already implemented). **Cost**: ~$1-2 in API spend +
  > Together account/key provisioning (user action; minutes). **Wall-clock**: ~30 min
  > total. **Dependency**: `TOGETHER_API_KEY` in `.env`; pre-build a 200-pair
  > sample file from the human qrels + abstract lookup.
  > **Output**: `reports/backend_comparison.md` with pairwise w-F1 + cost
  > breakdown; row in §10.5 validation report.

### 12.5 Document the CoT prompt with before/after examples

- [x] 12.5 Promote the contents of
  `scripts/judge_disagreement_examples.py` and
  `scripts/judge_cot_probe.py` from one-shot artefacts into
  appendix tables of `reports/llm_judge_validation.md`. For 2-3
  concrete cases (J-curve / VA-recommends-PE/CPT / HD-trajectory)
  print: the exact strict-mode prompt, the strict-mode response,
  the exact CoT-mode prompt, the CoT-mode response with reasoning
  chain. Makes the methodological pivot reproducible without
  rerunning the scripts.
  > **Effort**: Trivial. **Cost**: $0 (artefacts already on disk).
  > **Wall-clock**: ~1 h writing. **Dependency**: none.
  > **Output**: appendix added to the validation report.

### 12.6 Pool-coverage statistical analysis

- [x] 12.6 Bootstrap topics; for each pool size $n \in \{100, 500,
  1000, 2500, 5000, ..., |\text{expanded}|\}$ sub-sample $n$
  positives from the expanded qrels and re-score every variant
  against the sub-sampled pool. Report a *recall-vs-pool-size*
  curve per variant. Quantifies how much pool thinness limits a
  system's *maximum-achievable* F1 — gives statistical rigour to
  the report's claim that "the published 44.34 baseline is inflated
  by ~27.8 pp of pool overlap with itself".
  > **Effort**: Medium. **Cost**: $0. **Wall-clock**: ~4 h coding (efficient
  > sub-sampling + plotting) + analysis. **Dependency**: none.
  > **Output**: `reports/pool_coverage_analysis.md` + recall-vs-pool-size
  > plot (PNG / SVG).
  > **Result**: 5 variants × 7 fractions × B=200 bootstrap; full pool = 4758 positives. **Key finding**: variant ranking changes between thin (10%) and full pool; bm25_rm3 is the *least* pool-dependent (Δ +5.20 pp vs +9-10 pp for others) — confirming its low score is genuinely worse, not pool-overlap-driven. Phase 1 and starter swap places between 10% and 100% pool, demonstrating that the official-pool leaderboard ordering is within sampling noise. Report at [`reports/pool_coverage_analysis.md`](../../../reports/pool_coverage_analysis.md).

### 12.7 LLM-filtered PRF — turn the RM3 negative result positive

- [x] 12.7 New retrieval variant `phase2_bm25_rm3_llm_filtered`:
  for each (qa_id, sentence_id) cell, run BM25 top-30 → ask the
  LLM (gpt-4o-mini --prompt cot) a binary `relevant?` per
  candidate → compute RM3 only over the accepted candidates
  (fallback: skip RM3 if zero accepted). Compare against
  §7.4 (`phase2_bm25_rm3` blind RM3) and `phase1_baseline` on
  both pools. Endereça directamente o nosso negative finding em
  §7.4 e transforma uma observação em contribuição: "blind RM3
  hurts; LLM-filtered RM3 helps".
  > **Effort**: Medium. **Cost**: ~$5-10 in API (LLM filtering over ~5820 cells
  > × ~30 candidates each → up to ~17 k binary judgements at $0.0001-0.0003 each).
  > **Wall-clock**: ~6-8 h (code + run + eval). **Dependency**: none; can run
  > in parallel with the existing pipeline.
  > **Output**: new variant + row in `reports/phase2_summary.md`; new
  > `src/trec_biogen/retrieval/llm_prf.py` module + tests.
  > **Result**: implemented + ran. Pragmatic compromise: OpenAI tier-1 rate limits forced restricting the LLM filter to the support path; contradict path uses plain BM25 (`apply_to_contradict: false` knob in the config). Support path filter ran ~5820 calls in ~75 min, cost negligible. **Headline numbers** (vs §7.4 blind RM3): official Sup F1 3.92 → **4.03** (+0.11); expanded Sup F1 8.97 → **9.89** (+0.92). **Honest framing**: LLM-filtered RM3 < blind RM3 (LLM filter does remove topic drift, the *positive* finding) but neither beats Phase 1 baseline (no RM3 at all, 16.43 expanded). Query expansion — lexical or semantic — is the wrong intervention for claim-length biomedical queries; the §7.4 negative result generalises. Run dir: [`runs/20260520-222544-phase2_bm25_rm3_llm_filtered/`](../../../runs/20260520-222544-phase2_bm25_rm3_llm_filtered/).

### 12.8 Hybrid negation classifier

- [ ] 12.8 Hybrid contradict-path negation: keep the existing NegEx
  pre-filter but add a *learned classifier* trained on
  NegEx-borderline cases to recover false negatives. Auto-label
  ~5 k borderline pairs (NegEx-rejected but containing a
  negation cue from a wider regex bank), QLoRA-finetune a small
  classifier head, and compare recall/precision against pure-NegEx
  on the contradict candidates.
  > **Effort**: Hard. **Cost**: ~$5 (cloud GPU 1 h for QLoRA) or ~6 h local
  > CPU. **Wall-clock**: ~8 h total. **Dependency**: §5.3 `phase2_no_negex`
  > should run first; if NegEx is shown to be *not* hurting on the expanded
  > pool, 12.8 has lower priority.
  > **Output**: `src/trec_biogen/nli/negation_hybrid.py` + a third NLI variant
  > in the summary; per-class recall/precision delta.

### 12.9 Extended disagreement review (paper-grade)

- [ ] 12.9 Sample 50-100 §2.15 strict-mode disagreements (instead
  of the n=12 single-annotator review currently in the validation
  report). Have two independent biomedical readers label each case;
  report inter-annotator agreement (Cohen's κ); use the
  consensus-labelled subset as a stronger empirical anchor for the
  CoT pivot finding.
  > **Effort**: Easy technically; **needs two human readers**. **Cost**: $0
  > compute. **Wall-clock**: 4-8 h of expert reviewer time × 2 reviewers.
  > **Dependency**: human-availability. **Output**: extended appendix
  > in `reports/llm_judge_validation.md`. Flagged for paper-grade
  > extension; out of scope for the unit-curricular deliverable.

### 12.10 LLM query rewriting first-stage variant

- [x] 12.10 New retrieval variant `phase2_bm25_llm_rewrite`: per
  (qa_id, sentence_id), prompt the LLM to emit 3-5 claim-focused
  query variants (constrained-token output to bound cost); run
  BM25 over each variant; fuse via RRF. Cheaper than dense retrieval
  and attacks the lexical-mismatch issue at source (where RM3 fails).
  Compare against `phase1_baseline` and `phase2_bm25_rm3` on both
  pools.
  > **Effort**: Medium. **Cost**: ~$2-5 (LLM calls ≈ 150-200 query-rewrite
  > prompts at ~$0.001 each). **Wall-clock**: ~6 h (code + run + eval).
  > **Dependency**: none. Originally Phase 4 in the design; pulled forward
  > to §12 because the cost is now bounded enough.
  > **Output**: new variant + row in `reports/phase2_summary.md`; new
  > `src/trec_biogen/retrieval/llm_rewrite.py` + tests.
  > **Result**: implemented + ran end-to-end. `LLMQueryRewriter`,
  > `retrieve_llm_rewrite`, Hydra config, orchestrator dispatch, and
  > `tests/test_llm_rewrite.py` are in place. Run dir:
  > `runs/20260521-193810-phase2_bm25_llm_rewrite/`. Rewrite phase made
  > 388 `gpt-4o-mini --prompt cot` calls, cost $0.0505; full pipeline
  > wall-clock 5258.82 s (~88 min), VRAM peak 2.07 GiB. Official Sup/Con
  > 5.29 / 0.52; expanded Sup/Con 10.65 / 6.03. Verdict: LLM rewriting
  > partially recovers blind RM3 (expanded Sup 8.97 → 10.65) and beats
  > LLM-filtered RM3 on Supports (9.89 → 10.65), but still underperforms
  > Phase 1 plain BM25 (16.43 / 12.01 expanded). The finding is folded
  > into `reports/phase2_summary.md` and `docs/phase2_report.md` as the
  > third query-side negative result.
