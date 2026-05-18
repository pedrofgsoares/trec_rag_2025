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
