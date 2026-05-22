#!/usr/bin/env bash
# Phase 2 §9.2 — encode top-5M PMIDs with MedCPT-Article-Encoder.
#
# Wall-clock is heavily hardware-dependent (BERT-base, seq=512). Reference
# points measured on this project's dev machine:
#   - Quadro T1000 Max-Q, fp32 batch=8 :: ~6 docs/s → ~12 d for 5M  (no tensor cores)
#   - Quadro T1000 Max-Q, fp16 batch=32:: ~1.4 docs/s — fp16 *hurts* without tensor cores
#   - i7-10750H CPU, batch=8           :: ~4.5 docs/s → ~13 d for 5M
# For a real run, target a server-class GPU with tensor cores (A100/H100/L4) and
# enable --fp16 to get the design's ~24 h budget; otherwise expect days, not hours.
#
# Resumable: the Python module skips shards whose .npy already exists, so a
# crashed run picks up where it stopped on re-invocation.
#
# Inputs (default paths; override via env vars):
#   PMIDS_PARQUET    data/interim/top5m_pmids.parquet    (output of §9.1)
#   BM25_INDEX_DIR   data/indexes/pubmed_bm25            (abstract text source)
#
# Outputs:
#   $OUT_DIR/index.faiss          IndexFlatIP over normalised 768-d vectors
#   $OUT_DIR/pmid_lookup.parquet  row -> PMID lookup, written last (success marker)
#   $OUT_DIR/build_progress.jsonl one line per shard completed

set -euo pipefail

PMIDS_PARQUET="${PMIDS_PARQUET:-data/interim/top5m_pmids.parquet}"
BM25_INDEX_DIR="${BM25_INDEX_DIR:-data/indexes/pubmed_bm25}"
OUT_DIR="${OUT_DIR:-data/indexes/medcpt_5m}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MODEL="${MODEL:-ncbi/MedCPT-Article-Encoder}"
DEVICE="${DEVICE:-auto}"
# fp16 default is OFF because non-tensor-core GPUs (T1000, GTX 16xx, older mobile
# Turing) emulate fp16 ops at fp32 speed plus conversion overhead, which can be
# 3–5× slower than plain fp32. Enable FP16=1 only on A100/H100/L4/4070+/etc.
FP16="${FP16:-0}"

if [[ ! -f "$PMIDS_PARQUET" ]]; then
    echo "error: PMIDS_PARQUET not found at $PMIDS_PARQUET" >&2
    echo "       run: uv run python scripts/select_top5m_pmids.py first" >&2
    exit 1
fi

if [[ ! -d "$BM25_INDEX_DIR" ]]; then
    echo "error: BM25_INDEX_DIR not found at $BM25_INDEX_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# Pyserini 0.43 instantiates an OpenAI client at module import time
# (pyserini.encode._openai). We don't use OpenAI; a placeholder satisfies it.
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-used-by-this-pipeline}"

FP16_FLAG=""
if [[ "$FP16" == "1" || "$FP16" == "true" ]]; then
    FP16_FLAG="--fp16"
fi

uv run python -m trec_biogen.retrieval.build_dense \
    --pmids       "$PMIDS_PARQUET" \
    --out-dir     "$OUT_DIR" \
    --bm25-index  "$BM25_INDEX_DIR" \
    --model       "$MODEL" \
    --batch-size  "$BATCH_SIZE" \
    --device      "$DEVICE" \
    $FP16_FLAG
