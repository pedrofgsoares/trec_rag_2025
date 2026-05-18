## ADDED Requirements

### Requirement: Official BioGEN 2025 corpus is downloaded and verified
The system SHALL provide a script that downloads `biogen-2025-document-collection.zip` from the official URL, extracts it under `data/raw/pubmed_baseline/`, and verifies its integrity by document count.

#### Scenario: Successful download and verification
- **WHEN** the operator runs `scripts/download_pubmed.sh`
- **THEN** the archive is fetched, extracted, and the script reports a doc count of 26,805,982 (±0.1%) before exiting 0

#### Scenario: Corrupt or partial archive aborts
- **WHEN** the extracted document count differs from the expected value by more than 0.1%
- **THEN** the script exits non-zero, retains the partial download for inspection, and instructs the operator to re-run

### Requirement: Corpus is parsed to a stable JSONL form
The system SHALL parse the raw corpus into JSONL where each record carries `pmid`, `title`, `abstract`, `mesh` (list), `pubdate`, and `journal`. Records with missing or empty `abstract` SHALL be retained but flagged so they can be excluded from indexing on demand.

#### Scenario: Parse produces one JSONL line per PMID
- **WHEN** the parse step finishes
- **THEN** the line count of the output JSONL equals the corpus document count

#### Scenario: Records with empty abstracts are flagged
- **WHEN** a source record has no abstract
- **THEN** the JSONL record carries `"abstract": ""` AND `"empty_abstract": true`

### Requirement: BM25 Lucene index is built via Pyserini
The system SHALL build a single Pyserini Lucene index over the parsed JSONL with stored fields sufficient to fetch `pmid`, `title`, and `abstract` at retrieval time. The index SHALL be reproducible from the parsed JSONL via `scripts/build_indexes.sh`.

#### Scenario: Build runs to completion
- **WHEN** the operator runs `scripts/build_indexes.sh` against the parsed JSONL
- **THEN** the script writes a complete index under `data/indexes/bm25_pubmed/` and reports the indexed document count

#### Scenario: Index round-trips a known PMID
- **WHEN** Pyserini is queried for the title of a sentinel PMID known to be in the corpus
- **THEN** the returned title matches the JSONL record exactly

### Requirement: Retrieval API exposes both pool depths
The system SHALL expose a Python function that takes a query string and a pool size `k`, queries the BM25 index, and returns a ranked list of `(pmid, score)` tuples. The function MUST support `k = 100` and `k = 1000` in the same process without re-opening the index.

#### Scenario: Two-depth queries on a single index handle
- **WHEN** the support path requests `k=100` and the contradiction path subsequently requests `k=1000` on the same query
- **THEN** both calls succeed and the second is a strict superset of the first up to rank 100

### Requirement: Index integrity is checked on first use per run
The pipeline SHALL verify, before phase 1, that the index exists and exposes the expected document count. If the check fails, the pipeline SHALL exit before any retrieval work.

#### Scenario: Missing index aborts before retrieval
- **WHEN** `data/indexes/bm25_pubmed/` is missing or empty
- **THEN** the pipeline exits non-zero and prints the path that was missing
