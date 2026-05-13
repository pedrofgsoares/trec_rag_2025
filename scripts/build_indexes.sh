#!/usr/bin/env bash
# Build the Pyserini Lucene BM25 index over the BioGen 2025 PubMed corpus.
#
# Phases:
#   1. parse XML.gz -> parsed/pubmed.jsonl  (rich schema; task 3.3/3.5)
#   2. convert      -> collection/pubmed.jsonl  ({id, contents}; task 4.1)
#   3. pyserini index --storeDocvectors --storeRaw  (task 4.1, 4.2)
#
# Long-running: ~12 h on the target laptop. Run overnight, capture wall-clock.
#
# Task: 4.1, 4.2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw/pubmed_baseline"
PARSED="${REPO_ROOT}/data/interim/pubmed.jsonl"
COLLECTION_DIR="${REPO_ROOT}/data/interim/collection"
INDEX_DIR="${REPO_ROOT}/data/indexes/pubmed_bm25"
LOG_DIR="${REPO_ROOT}/runs/index_build"
THREADS="${PYSERINI_THREADS:-10}"  # leave 2 cores headroom on 12-thread CPU

mkdir -p "$(dirname "${PARSED}")" "${COLLECTION_DIR}" "${INDEX_DIR}" "${LOG_DIR}"

if [[ ! -f "${PARSED}" ]]; then
    echo "[1/3] parsing PubMed XML -> ${PARSED}"
    time python -m trec_biogen.ingest.parse_pubmed --input "${RAW_DIR}" --output "${PARSED}"
else
    echo "[1/3] parsed JSONL already present: ${PARSED}"
fi

COLL_FILE="${COLLECTION_DIR}/pubmed.jsonl"
if [[ ! -f "${COLL_FILE}" ]]; then
    echo "[2/3] converting -> ${COLL_FILE}"
    time python -m trec_biogen.retrieval.build_collection \
        --input "${PARSED}" --output "${COLL_FILE}"
else
    echo "[2/3] collection JSONL already present: ${COLL_FILE}"
fi

START_TS=$(date +%s)
LOG_FILE="${LOG_DIR}/index_$(date +%Y%m%d_%H%M%S).log"
echo "[3/3] building Lucene index -> ${INDEX_DIR}  (log: ${LOG_FILE})"

# Pyserini wraps Anserini Lucene indexer. Threads pinned for headroom.
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "${COLLECTION_DIR}" \
    --index "${INDEX_DIR}" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${THREADS}" \
    --storePositions --storeDocvectors --storeRaw \
    2>&1 | tee "${LOG_FILE}"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
SIZE=$(du -sh "${INDEX_DIR}" | cut -f1)
echo "[done] wall-clock=${ELAPSED}s size=${SIZE}"
echo "wall_clock_seconds=${ELAPSED}" >> "${LOG_FILE}"
echo "index_size=${SIZE}" >> "${LOG_FILE}"
