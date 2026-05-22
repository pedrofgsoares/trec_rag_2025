"""Pick the top-5M PMIDs by a citation-frequency proxy (Phase 2 §9.1).

Pyserini does not expose a real citation count for PubMed abstracts at
index time, so we approximate it by **document length** (the sum of
term frequencies over the stored document vector):

    score(d) = sum_t  tf(t, d)

Substantial PubMed abstracts (richer methods, more entities, longer
discussion) score above stubby or boilerplate entries. Doc length is
the strongest free signal we have at index time for "useful as a
retrieval target".

Implementation notes (the v2 rewrite — chunked, restart-resumable):

The naive heap-based top-N selection that this script originally used
saturated RAM on a 12 GB host after ~12 h: a 5M-entry Python heap +
JVM heap growth + mmap'd index working set forced 5+ GB into swap and
the kernel began thrashing (~700 GB of paged reads). Throughput
collapsed.

The v2 design writes per-chunk Parquets to a cache dir and does the
top-N selection as a single merge-sort over the union of those files
at the end. Peak RAM stays under ~1 GB during scoring (one chunk +
Pyserini's working set), and the final merge holds the full 26.8M
``(pmid, score)`` rows in memory (~540 MB) only briefly.

The chunk cache also makes the script restart-resumable: a chunk file
that already exists with non-zero size is skipped on re-invocation.

Usage::

    uv run python scripts/select_top5m_pmids.py \\
        --index data/indexes/pubmed_bm25 \\
        --top-n 5000000 \\
        --out   data/interim/top5m_pmids.parquet
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

DEFAULT_CHUNK_SIZE = 500_000     # ~25 MB per chunk parquet
DEFAULT_CACHE_DIR = Path("data/interim/_select_top5m_chunks")


def _open_reader(index_dir: Path):
    # Pyserini 0.43 renamed the class; the older ``IndexReader`` alias is gone.
    from pyserini.index.lucene import LuceneIndexReader

    return LuceneIndexReader(str(index_dir))


def score_chunk(
    reader,
    *,
    start: int,
    end: int,
    out_path: Path,
) -> int:
    """Score docs ``[start, end)`` and write ``(pmid, score)`` to ``out_path``.

    Returns the number of rows written. The chunk is kept entirely in
    Python lists; peak RAM is ~25 MB for a 500k-doc chunk.
    """
    pmids: list[str] = []
    scores: list[float] = []
    for i in range(start, end):
        pmid = reader.convert_internal_docid_to_collection_docid(i)
        if pmid is None:
            continue
        tv = reader.get_document_vector(pmid)
        score = float(sum(tv.values())) if tv else 0.0
        pmids.append(pmid)
        scores.append(score)
    pl.DataFrame({"pmid": pmids, "score": scores}).write_parquet(out_path)
    return len(pmids)


def merge_top_n(chunk_paths: list[Path], *, top_n: int) -> pl.DataFrame:
    """Concatenate chunk parquets, sort by score descending, return top-N."""
    parts = [pl.read_parquet(p) for p in chunk_paths]
    merged = pl.concat(parts)
    return merged.sort("score", descending=True).head(top_n)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="select_top5m_pmids")
    p.add_argument("--index", type=Path, default=Path("data/indexes/pubmed_bm25"))
    p.add_argument("--top-n", type=int, default=5_000_000)
    p.add_argument("--out", type=Path, default=Path("data/interim/top5m_pmids.parquet"))
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help="Per-chunk parquet cache. Existing non-empty chunks are skipped on re-run.")
    p.add_argument("--start-chunk", type=int, default=None,
                   help="If set, process only chunks in [start-chunk, end-chunk] (inclusive). "
                        "Used by the subprocess-per-chunk wrapper to bound JVM heap growth.")
    p.add_argument("--end-chunk", type=int, default=None)
    p.add_argument("--no-merge", action="store_true",
                   help="Skip the final merge step (useful when running one chunk per subprocess).")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip scoring; only run the merge over already-cached chunks.")
    args = p.parse_args(argv)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    # Merge-only path: skip the index entirely.
    if args.merge_only:
        chunk_paths = sorted(args.cache_dir.glob("chunk_*.parquet"))
        if not chunk_paths:
            print(f"error: no chunks found in {args.cache_dir}", file=sys.stderr)
            return 2
        print(f"[select_top5m] merge-only over {len(chunk_paths)} chunks → top-{args.top_n}", flush=True)
        final_df = merge_top_n(chunk_paths, top_n=args.top_n)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(args.out)
        print(
            f"[select_top5m] wrote {args.out} ({final_df.shape[0]} pmids) "
            f"in {(time.monotonic() - t_start)/60:.1f}m",
            flush=True,
        )
        return 0

    reader = _open_reader(args.index)
    stats = reader.stats()
    n_docs = stats.get("documents") or 0
    if n_docs == 0:
        print("error: index reports 0 documents", file=sys.stderr)
        return 2

    chunks: list[tuple[int, int, Path]] = []
    for chunk_idx, start in enumerate(range(0, n_docs, args.chunk_size)):
        end = min(start + args.chunk_size, n_docs)
        path = args.cache_dir / f"chunk_{chunk_idx:05d}.parquet"
        chunks.append((start, end, path))

    # Restrict to the requested chunk range (subprocess-per-chunk mode).
    if args.start_chunk is not None:
        lo = args.start_chunk
        hi = args.end_chunk if args.end_chunk is not None else lo
        chunks_to_consider = [
            (i, c) for i, c in enumerate(chunks) if lo <= i <= hi
        ]
    else:
        chunks_to_consider = list(enumerate(chunks))

    print(
        f"[select_top5m] index has {n_docs} documents; "
        f"{len(chunks)} total chunks; processing {len(chunks_to_consider)} this invocation",
        flush=True,
    )

    skipped = 0
    for chunk_idx, (start, end, path) in chunks_to_consider:
        if path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        t0 = time.monotonic()
        n_written = score_chunk(reader, start=start, end=end, out_path=path)
        dt = time.monotonic() - t0
        rate = n_written / dt if dt > 0 else 0.0
        elapsed = time.monotonic() - t_start
        remaining = len(chunks) - chunk_idx - 1
        eta_min = remaining * dt / 60 if dt > 0 else 0
        print(
            f"[chunk {chunk_idx + 1:>3}/{len(chunks)}] "
            f"docs {start:>10}-{end:>10}: {n_written:>7} written in {dt:>6.1f}s "
            f"({rate:>5.0f} docs/s)  elapsed={elapsed/60:5.1f}m  eta≈{eta_min:5.1f}m",
            flush=True,
        )
        gc.collect()

    if skipped:
        print(f"[select_top5m] skipped {skipped} cached chunks", flush=True)

    if args.no_merge:
        print(f"[select_top5m] --no-merge: exiting without final merge", flush=True)
        return 0

    # Default behaviour: only merge when every chunk is on disk.
    missing = [p for _, _, p in chunks if not (p.exists() and p.stat().st_size > 0)]
    if missing:
        print(
            f"[select_top5m] {len(missing)} chunks still missing — skipping merge "
            f"(re-run without --start-chunk to finish, or use --merge-only when done)",
            flush=True,
        )
        return 0

    print(f"[select_top5m] merging {len(chunks)} chunks → top-{args.top_n}", flush=True)
    final_df = merge_top_n([p for _, _, p in chunks], top_n=args.top_n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(args.out)
    total_time = time.monotonic() - t_start
    print(
        f"[select_top5m] wrote {args.out} ({final_df.shape[0]} pmids) "
        f"in {total_time/60:.1f}m total",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
