## ADDED Requirements

### Requirement: Pipeline accepts the official Task A input JSONL
The system SHALL read an input file conforming to the official BioGEN 2025 Task A format: UTF-8 JSONL where each line is a topic object containing a `metadata` block (`team_id`, `run_id`, `qa_id`, `question`) and an `answer` array of sentence objects. Each sentence object SHALL be addressable by its `qa_id` and its position within the `answer` array.

#### Scenario: Valid input file is loaded
- **WHEN** the pipeline is invoked with `--input` pointing to a valid Task A JSONL file
- **THEN** the system loads every topic, exposes per-sentence iteration, and logs the count of topics and sentences

#### Scenario: Malformed input fails fast
- **WHEN** any line is missing `metadata.qa_id` or `answer`
- **THEN** the system aborts before retrieval with a non-zero exit code and a message naming the offending line number

### Requirement: Pipeline produces a valid Task A submission JSONL
The system SHALL write a submission file in the official BioGEN 2025 Task A format. For every input sentence, the corresponding output sentence object SHALL contain the original `text`, an `existing_supported_citations` array carrying any input-provided PMIDs unchanged, a `supported_citations` array of at most 3 newly assigned supporting PMIDs, and a `contradicted_citations` array of at most 3 contradicting PMIDs.

#### Scenario: Caps are enforced per sentence
- **WHEN** the pipeline writes any sentence
- **THEN** `len(supported_citations) ≤ 3` AND `len(contradicted_citations) ≤ 3`

#### Scenario: All assigned PMIDs come from the official corpus
- **WHEN** any PMID appears in `supported_citations` or `contradicted_citations`
- **THEN** that PMID exists in the indexed BioGEN 2025 PubMed snapshot

#### Scenario: Submission preserves topic and sentence order
- **WHEN** the pipeline writes the submission
- **THEN** topics appear in input order and sentences within each topic appear in input order

### Requirement: Decoupled support and contradiction retrieval paths
The system SHALL execute two independent retrieval paths sharing one BM25 index. The support path SHALL retrieve the top-100 documents per (question + sentence) query. The contradiction path SHALL retrieve the top-1000 documents per query. Neither path may reuse the other's pool.

#### Scenario: Both paths run with their distinct k values
- **WHEN** the pipeline runs end to end
- **THEN** `retrieval_support.parquet` contains exactly 100 rows per sentence and `retrieval_contradict.parquet` contains exactly 1000 rows per sentence (or fewer only if BM25 returned fewer hits)

### Requirement: Sentence-level NLI on the contradiction path
The contradiction path SHALL segment each candidate abstract into sentences before NLI. The NLI model SHALL receive `(answer_sentence, abstract_sentence)` pairs and produce a per-pair contradiction probability. The per-document contradiction score SHALL be the max over its constituent sentences.

#### Scenario: Abstracts are segmented before classification
- **WHEN** any candidate abstract enters the contradiction NLI step
- **THEN** it is segmented by scispaCy into one or more sentences and each sentence is scored individually

#### Scenario: Aggregation uses max-pooling
- **WHEN** an abstract has N segmented sentences
- **THEN** the document-level contradiction score equals `max` of the N per-sentence contradiction probabilities

### Requirement: NegEx pre-filter on the contradiction path
Before contradiction NLI runs, the system SHALL drop candidate abstract sentences that contain no negation cue. The cue set SHALL be the union of `negspacy` defaults and the explicit biomedical cue list defined in `design.md` (D4).

#### Scenario: Sentence with negation cue is kept
- **WHEN** an abstract sentence contains "no evidence of"
- **THEN** the sentence is passed to the NLI step

#### Scenario: Sentence without negation cue is dropped
- **WHEN** an abstract sentence contains no cue from the configured list
- **THEN** the sentence is excluded from NLI and the drop is logged in `filtered_out_count`

### Requirement: Selection respects caps, ordering, and global dedup
The selection module SHALL emit, per sentence, up to 3 contradicting PMIDs followed by up to 3 supporting PMIDs (contradicting first). Within a topic, no PMID may appear in more than one sentence's combined output across both classes.

#### Scenario: Contradicting PMIDs precede supporting PMIDs in the file
- **WHEN** a sentence has both contradicting and supporting candidates
- **THEN** the submission writer orders contradicting first per the track rule

#### Scenario: Duplicate PMIDs across sentences are removed
- **WHEN** a PMID is selected for sentence A and would also be selected for sentence B in the same topic
- **THEN** it is kept on sentence A (lower index) and dropped from sentence B; if dropping leaves B empty in that class, the next-best candidate is promoted

### Requirement: Sequential model loading within VRAM budget
The pipeline SHALL hold at most one heavy model in GPU memory at any time. Before phase N+1 loads its model, phase N's model SHALL be released and CUDA cache emptied. The pipeline SHALL refuse to start if measured free VRAM is below 3.5 GiB or measured RAM is below 11 GiB.

#### Scenario: Preflight rejects under-provisioned environment
- **WHEN** the pipeline starts and `psutil.virtual_memory().total < 11 GiB` OR free VRAM `< 3.5 GiB`
- **THEN** the pipeline exits non-zero before phase 1 with an actionable error message

#### Scenario: Models are released between phases
- **WHEN** any phase completes
- **THEN** its model object is deleted and `torch.cuda.empty_cache()` is called before the next phase begins

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
