#!/usr/bin/env bash
# Wrapper that runs select_top5m_pmids.py one chunk per fresh Python+JVM
# subprocess. Each invocation processes a single chunk and exits, so the
# JVM heap cannot grow without bound across the 54 chunks. ~5s JVM
# startup × 54 chunks = ~5 min overhead, comfortably under the wall-clock
# saved by avoiding swap thrashing.
#
# Usage:
#   bash scripts/select_top5m_loop.sh        # process all chunks then merge
#   N_CHUNKS=54 BATCH=1 bash scripts/select_top5m_loop.sh
#
# Idempotent: existing non-empty chunk parquets are skipped. Safe to re-run
# after an interruption — pick up where it stopped.

set -euo pipefail

N_CHUNKS="${N_CHUNKS:-54}"
BATCH="${BATCH:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-500000}"
INDEX="${INDEX:-data/indexes/pubmed_bm25}"
TOP_N="${TOP_N:-5000000}"
CACHE_DIR="${CACHE_DIR:-data/interim/_select_top5m_chunks}"
OUT="${OUT:-data/interim/top5m_pmids.parquet}"
LOG="${LOG:-data/interim/_select_top5m.log}"

mkdir -p "$(dirname "$LOG")" "$CACHE_DIR"
echo "=== select_top5m loop: started $(date -u +%H:%M:%S) UTC ===" | tee -a "$LOG"
echo "n_chunks=$N_CHUNKS  batch=$BATCH  chunk_size=$CHUNK_SIZE" | tee -a "$LOG"

# Iterate chunks in batches of $BATCH. Each iteration spawns a fresh JVM.
# Start at 0 even if early chunks are cached — they'll be skipped quickly.
i=0
while [ "$i" -lt "$N_CHUNKS" ]; do
    end=$(( i + BATCH - 1 ))
    [ "$end" -ge "$N_CHUNKS" ] && end=$(( N_CHUNKS - 1 ))
    echo "--- subprocess: chunks $i..$end ---" | tee -a "$LOG"
    uv run --quiet python scripts/select_top5m_pmids.py \
        --index       "$INDEX" \
        --top-n       "$TOP_N" \
        --chunk-size  "$CHUNK_SIZE" \
        --cache-dir   "$CACHE_DIR" \
        --out         "$OUT" \
        --start-chunk "$i" \
        --end-chunk   "$end" \
        --no-merge 2>&1 | tee -a "$LOG"
    i=$(( end + 1 ))
done

echo "=== all chunks done; running final merge ===" | tee -a "$LOG"
uv run --quiet python scripts/select_top5m_pmids.py \
    --top-n     "$TOP_N" \
    --cache-dir "$CACHE_DIR" \
    --out       "$OUT" \
    --merge-only 2>&1 | tee -a "$LOG"

echo "=== done $(date -u +%H:%M:%S) UTC; output=$OUT ===" | tee -a "$LOG"
