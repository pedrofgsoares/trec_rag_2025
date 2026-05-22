## ADDED Requirements

### Requirement: Architectural variants selectable via Hydra
The system SHALL expose at least six named Phase 2 variants of the pipeline, each as a Hydra run config under `configs/run/phase2_<variant>.yaml`. Each variant SHALL be invocable end-to-end with a single command and SHALL produce its own `runs/<id>/` directory populated with the same intermediate Parquet schema as the Phase 1 pipeline.

The six minimal variants SHALL be:

| Name | Architectural change vs Phase 1 |
|---|---|
| `phase2_no_rerank` | Skips the MedCPT-CE rerank phase; the support path emits BM25 top-3 directly |
| `phase2_no_negex` | Skips the NegEx cue-list filter; the contradict NLI runs on every segmented abstract sentence |
| `phase2_allow_existing` | Disables exclusion of `existing_supported_citations` from new picks |
| `phase2_scifive_large` | Replaces the contradict-path NLI model with `razent/SciFive-large-Pubmed_PMC-MedNLI` (fp16, chunk=4) |
| `phase2_bm25_rm3` | Enables Pyserini RM3 query expansion on both retrieval phases |
| `phase2_hybrid` | First-stage retrieval is RRF fusion of BM25 and a MedCPT-Article-Encoder FAISS-CPU index over the top-5M-by-citation-frequency subset |

#### Scenario: Each named variant runs end-to-end
- **WHEN** the operator runs `python -m trec_biogen.pipeline.run_task_a --config-name phase2_<variant>` for any of the six names
- **THEN** the pipeline produces a `task_a_output.json` that passes `validate_official` and a `metadata.yaml` recording the variant name and the resolved config

#### Scenario: Variants share intermediate Parquets via resume mode
- **WHEN** a variant changes only a downstream phase (e.g., `phase2_no_negex` only changes the contradict NLI input) and the operator invokes it with `BIOGEN_RUN_DIR=<phase1-run-dir>` and `--reuse-from=<phase1-run-dir>`
- **THEN** unchanged upstream Parquets (retrieval, segmentation) are reused; the run completes faster than a cold start

### Requirement: Variant compositions are valid Hydra invocations
The system SHALL allow combining variant overrides on a single CLI invocation so the operator can run e.g. `phase2_scifive_large` plus `no_negex` without writing a new config file.

#### Scenario: Composed variant runs
- **WHEN** the operator runs `python -m trec_biogen.pipeline.run_task_a --config-name phase2_scifive_large nli.contradict.negex=false`
- **THEN** the pipeline runs with both overrides applied and `metadata.yaml` records the full resolved config so the run is reproducible

### Requirement: Hybrid retrieval index over the top-5M subset
The system SHALL provide a script that selects the top 5,000,000 PubMed documents by citation frequency (proxied by Lucene term-document frequency from the existing BM25 index) and encodes them using `ncbi/MedCPT-Article-Encoder` into a FAISS-CPU index stored under `data/indexes/medcpt_5m/`. The encoding SHALL be a one-off cost; subsequent `phase2_hybrid` runs SHALL load the pre-built index without re-encoding.

#### Scenario: Encoding script produces a loadable FAISS index
- **WHEN** the operator runs `bash scripts/build_dense_index.sh`
- **THEN** the script writes `data/indexes/medcpt_5m/index.faiss` + `data/indexes/medcpt_5m/pmid_lookup.parquet` and reports wall-clock and final size

#### Scenario: Hybrid retrieval fuses BM25 and dense via RRF
- **WHEN** the `phase2_hybrid` variant runs first-stage retrieval
- **THEN** for each (qa_id, sentence_id) it issues k=1000 BM25 hits and k=1000 FAISS hits, fuses via Reciprocal Rank Fusion with parameter `k=60`, and writes the fused top-1000 to `retrieval_contradict.parquet` for the downstream pipeline

### Requirement: Progress instrumentation on long phases
The pipeline SHALL emit `tqdm`-style progress bars during the abstract segmentation phase (`phases.segment_abstracts`) and the NegEx filter phase (`nli.negation.filter_negated`), updating at least once per 1% of total work, so that wall-clock progress is visible during the multi-minute-to-multi-hour passes.

#### Scenario: Operator can see segmentation progress
- **WHEN** `phases.segment_abstracts` runs over ≥ 10,000 unique PMIDs
- **THEN** a `tqdm` bar advances at least once per percentage of total PMIDs processed

#### Scenario: Operator can see NegEx progress
- **WHEN** `nli.negation.filter_negated` runs over ≥ 100,000 segmented sentences
- **THEN** a `tqdm` bar advances at least once per percentage of total sentences scanned

### Requirement: Per-run resource accounting captured in metadata
The `metadata.yaml` produced by every pipeline run SHALL include `wall_clock_seconds_per_phase` (dict keyed by phase name) and `vram_peak_gb_per_phase` (dict, populated from `torch.cuda.max_memory_allocated()` reset between phases). These fields enable a Pareto-frontier comparison across variants.

#### Scenario: Phase-level timings are recorded
- **WHEN** any pipeline run finishes
- **THEN** `metadata.yaml` contains a `wall_clock_seconds_per_phase` dict with one entry per executed phase (entries SHALL NOT appear for skipped phases reused via `_maybe_run()`)

#### Scenario: Phase-level VRAM peaks are recorded
- **WHEN** any pipeline phase loads a CUDA model
- **THEN** the peak VRAM observed during that phase is recorded in `vram_peak_gb_per_phase` under the same key as the wall-clock timing
