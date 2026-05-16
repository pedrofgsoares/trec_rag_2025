#!/usr/bin/env bash
# Build the Pyserini Lucene BM25 index over the BioGen 2025 PubMed corpus.
#
# The official corpus is distributed in Pyserini JsonCollection shape
# ({"id","contents"} per line) under a subfolder of RAW_DIR (typically
# named "pubmed_baseline_collection_jsonl") — so we can point Pyserini
# directly at the unzipped data, no parse/convert step required.
#
# Long-running: ~12 h on the target laptop. Run overnight, capture wall-clock.
#
# Task: 4.1, 4.2

set -euo pipefail

# Pyserini 0.43 needs OPENAI_API_KEY at module load time (see comment in
# scripts/run_starter_baseline.sh). Placeholder satisfies the constructor.
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-used-by-this-pipeline}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Same override as download_pubmed.sh — raw corpus may live on /mnt/c/...
RAW_DIR="${BIOGEN_RAW_DIR:-${REPO_ROOT}/data/raw/pubmed_baseline}"
# INDEX_DIR can be overridden when the WSL VHD lacks space, e.g.
#   BIOGEN_INDEX_DIR=/mnt/c/Users/<you>/Downloads/TREC/indexes/pubmed_bm25
# Trade-off: /mnt/c is ~3-5× slower than native ext4 for the heavy random
# I/O of Lucene indexing, but is necessary if WSL is disk-bound.
INDEX_DIR="${BIOGEN_INDEX_DIR:-${REPO_ROOT}/data/indexes/pubmed_bm25}"
LOG_DIR="${REPO_ROOT}/runs/index_build"

# Memory tuning for the Lucene indexer on the 8 GB WSL2 VM.
#
# Total budget (measured 2026-05-14, free -h): 7.6 GiB VM, 5.9 GiB available.
# Reserve ~2.2 GiB for kernel + page cache + Python + JVM off-heap, leaves ~4 GiB
# safe headroom for the JVM heap. -Xmx4g + 2 threads keeps the total resident
# set comfortably under the VM ceiling so the kernel never OOM-kills the JVM.
#
# If you bump .wslconfig to 12 GB later (task 1.1) and verify with `free -h`,
# override at invocation:
#   _JAVA_OPTIONS="-Xmx8g -Xms2g" PYSERINI_THREADS=4 bash scripts/build_indexes.sh
export _JAVA_OPTIONS="${_JAVA_OPTIONS:--Xmx4g -Xms1g}"
THREADS="${PYSERINI_THREADS:-2}"

mkdir -p "${INDEX_DIR}" "${LOG_DIR}"

# Locate the directory that actually contains the JSONL files. Pyserini's
# JsonCollection needs a directory of .jsonl files, not a single file.
JSONL_FILE=$(find "${RAW_DIR}" -type f -name '*.jsonl' -print -quit)
if [[ -z "${JSONL_FILE}" ]]; then
    echo "ERROR: no .jsonl files found under ${RAW_DIR}" >&2
    echo "       run scripts/download_pubmed.sh first" >&2
    exit 2
fi
COLLECTION_DIR="$(dirname "${JSONL_FILE}")"
JSONL_COUNT=$(find "${COLLECTION_DIR}" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
echo "[input] collection dir: ${COLLECTION_DIR}  (${JSONL_COUNT} jsonl files)"

START_TS=$(date +%s)
LOG_FILE="${LOG_DIR}/index_$(date +%Y%m%d_%H%M%S).log"
echo "[index] building Lucene index -> ${INDEX_DIR}  (log: ${LOG_FILE})"
echo "[index] threads=${THREADS}  _JAVA_OPTIONS='${_JAVA_OPTIONS}'"

# Pyserini wraps Anserini Lucene indexer. Threads pinned for headroom on the
# 12-thread i7-10750H. --storeRaw is required so the rerank/NLI phases can
# fetch document contents via BM25Index.doc_text().
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
