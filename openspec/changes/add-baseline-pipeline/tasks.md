## 1. Environment & Hardware Provisioning

- [ ] 1.1 Operator: bump WSL2 RAM to ≥12 GB via host `C:\Users\<user>\.wslconfig` (`[wsl2] memory=12GB swap=8GB`) and run `wsl --shutdown`
- [ ] 1.2 Verify post-restart: `free -h` reports ≥11 GiB total inside WSL
- [ ] 1.3 Install OpenJDK 21 (`sudo apt install openjdk-21-jdk-headless`) and confirm `java -version`
- [ ] 1.4 Install Python 3.11 via `pyenv` and create project venv with `uv venv`
- [x] 1.5 Verify `nvidia-smi` runs inside WSL and reports the Quadro T1000 with 4 GB

## 2. Project Scaffolding

- [ ] 2.1 Create directory tree: `src/trec_biogen/{ingest,retrieval,rerank,nli,pipeline,io,eval}/`, `configs/{retrieval,rerank,nli,run}/`, `data/{raw,interim,indexes,topics,qrels}/`, `runs/`, `notebooks/`, `scripts/`, `tests/`
- [ ] 2.2 Add `pyproject.toml` pinning: `pyserini`, `transformers`, `sentence-transformers`, `torch` (CUDA 12.1 wheel), `scispacy`, `negspacy`, `hydra-core`, `mlflow`, `duckdb`, `polars`, `pytest`
- [ ] 2.3 Install `en_core_sci_sm` from scispaCy releases
- [ ] 2.4 Add `.gitignore` excluding `data/`, `runs/`, `.hydra/`, `*.parquet`
- [ ] 2.5 Initialise `git` repo and make first commit of scaffolding

## 3. Corpus Ingestion (capability: pubmed-index)

- [ ] 3.1 Write `scripts/download_pubmed.sh` to fetch `biogen-2025-document-collection.zip` and extract under `data/raw/pubmed_baseline/`
- [ ] 3.2 Add post-download check that asserts doc count = 26,805,982 (±0.1%)
- [ ] 3.3 Implement `src/trec_biogen/ingest/parse_pubmed.py` to emit JSONL records `{pmid,title,abstract,mesh,pubdate,journal,empty_abstract}`
- [ ] 3.4 Add unit test on a 100-doc fixture verifying record schema and the `empty_abstract` flag
- [ ] 3.5 Run parse over the full corpus; confirm output line count equals corpus doc count

## 4. BM25 Index Build (capability: pubmed-index)

- [ ] 4.1 Write `scripts/build_indexes.sh` invoking Pyserini `index` with `JsonCollection`, `--storeDocvectors`, `--storeRaw`
- [ ] 4.2 Run build in background overnight; capture wall-clock and final size
- [ ] 4.3 Implement `src/trec_biogen/retrieval/bm25.py` exposing `search(query, k)` that supports `k=100` and `k=1000` on one open index handle
- [ ] 4.4 Add round-trip test: query a sentinel PMID's title and confirm exact retrieval
- [ ] 4.5 Add preflight `verify_index()` used by the pipeline entry point

## 5. Topic & Qrels Loading (capability: biogen-task-a)

- [ ] 5.1 Implement `src/trec_biogen/io/topics.py` to load the official Task A input JSONL and validate `metadata.qa_id` and `answer` presence
- [ ] 5.2 Add fail-fast on malformed lines with offending line number in the error
- [ ] 5.3 Place BioGen 2024 and 2025 qrels under `data/qrels/` and add a loader producing per-(qa_id, sentence_id, class) positive-set lookups
- [ ] 5.4 Add fixtures: 2-topic mini input + matching mini qrels for unit tests

## 6. Baseline Reproduction (capability: evaluation)

- [ ] 6.1 Vendor or git-submodule the `trec-biogen/starter-kit-2025` repo under `external/starter-kit-2025/`
- [ ] 6.2 Write `scripts/run_starter_baseline.sh` that runs the unmodified starter-kit baseline against the official 2025 input
- [ ] 6.3 Implement `src/trec_biogen/eval/metrics.py` computing per-class P/R/F1 under Strict and Relaxed settings
- [ ] 6.4 Implement `make baseline-check` (or `scripts/baseline_check.sh`) that runs starter baseline + eval and asserts F1 within ±2 of (44.34, 4.67)
- [ ] 6.5 Run the gate; do not proceed past this task until it exits 0

## 7. Support Path (capability: biogen-task-a)

- [ ] 7.1 Implement `src/trec_biogen/pipeline/run_task_a.py` phase 1 that issues BM25 k=100 per (question + sentence) and writes `runs/<id>/retrieval_support.parquet`
- [ ] 7.2 Implement `src/trec_biogen/rerank/cross_encoder.py` loading `ncbi/MedCPT-Cross-Encoder` with batch=8 and writing `runs/<id>/rerank_support.parquet` (top-30 per sentence)
- [ ] 7.3 Implement `src/trec_biogen/nli/stance.py` for support: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` over (sentence, title+abstract) truncated to 512 tokens, batch=16, writing `runs/<id>/nli_support.parquet`
- [ ] 7.4 Add explicit `del model; torch.cuda.empty_cache()` between phases
- [ ] 7.5 Spot-check on the 2-topic fixture that support scores are reasonable (top-1 entailment > 0.5 on at least one sentence)

## 8. Contradiction Path (capability: biogen-task-a)

- [ ] 8.1 Phase 1' issues BM25 k=1000 per (question + sentence) and writes `runs/<id>/retrieval_contradict.parquet`
- [ ] 8.2 Implement abstract sentence segmentation via scispaCy `en_core_sci_sm` writing one row per (candidate_pmid, abstract_sentence_idx, abstract_sentence_text)
- [ ] 8.3 Implement NegEx + cue-list filter (the 23 patterns from `design.md` D4) in `src/trec_biogen/nli/negation.py`; log `filtered_out_count`
- [ ] 8.4 Implement contradiction NLI in `nli/stance.py` using `razent/SciFive-base-Pubmed_PMC` fine-tuned on MedNLI, batch=8, writing per-pair contradiction probability
- [ ] 8.5 Aggregate per-document score by max-pool over abstract sentences; write `runs/<id>/nli_contradict.parquet`
- [ ] 8.6 Sample 50 NegEx-dropped sentences and manually review for false negatives; record findings in `runs/<id>/negation_audit.md`

## 9. Selection & Submission (capability: biogen-task-a)

- [ ] 9.1 Implement `src/trec_biogen/pipeline/selection.py`: per sentence, take top-3 supports by NLI score above τ_sup; top-3 contradictions by BM25 rank above τ_con
- [ ] 9.2 Apply global PMID dedup within a topic: lower-index sentence wins ties; promote next-best on the loser
- [ ] 9.3 Implement `src/trec_biogen/io/submission.py` writing JSONL with contradicting PMIDs first, then supporting PMIDs, preserving topic+sentence order
- [ ] 9.4 Add submission validator asserting per-sentence caps, PMID membership in the corpus, and ordering rule
- [ ] 9.5 Run end-to-end on the 2-topic fixture; confirm validator passes

## 10. Evaluation & Reporting (capability: evaluation)

- [ ] 10.1 Extend `eval/metrics.py` to emit JSON report with the six numbers × two settings against any qrels file
- [ ] 10.2 Add `eval/report.py` writing `report.md` with the leaderboard comparison table (baseline / CLaC / InfoLab / current run)
- [ ] 10.3 Add `phase1_pass` flag (Supports F1 ≥ 60 AND Contradicts F1 ≥ 10 strict, 2025 qrels)
- [ ] 10.4 Run full pipeline end-to-end against the official 2025 input and qrels; record numbers in `reports/exp_001_baseline_phase1.md`
- [ ] 10.5 Compare against published Table 5 rows; document deltas

## 11. Reproducibility & Run Hygiene

- [ ] 11.1 Wire Hydra config: every CLI snapshots resolved config + git SHA + hardware fingerprint into `runs/<id>/metadata.yaml`
- [ ] 11.2 Add structured logging via `loguru` with one JSONL log per run
- [ ] 11.3 Add MLflow local tracking; one run per pipeline invocation with all six metric numbers
- [ ] 11.4 Document setup, commands, and known limits in `README.md`
- [ ] 11.5 Add `pytest` smoke test that runs the 2-topic fixture end-to-end and asserts a valid submission is produced

## 12. Phase 1 Sign-Off

- [ ] 12.1 Verify all six metric numbers (Supports + Contradicts × Strict + Relaxed against 2024 + 2025 qrels) are recorded
- [ ] 12.2 Verify Phase-1 thresholds met OR write a `reports/phase1_gap_analysis.md` enumerating which threshold(s) missed and the proposed Phase-2 mitigation
- [ ] 12.3 Tag the repo `phase1-baseline` and archive the change via `/opsx:archive`
