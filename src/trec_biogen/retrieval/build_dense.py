"""Build the MedCPT-Article-Encoder FAISS index over the top-5M PubMed subset.

Phase 2 §9.2. CLI module invoked by ``scripts/build_dense_index.sh``.

Input:
  * ``--pmids`` parquet with one column ``pmid`` (output of §9.1
    ``scripts/select_top5m_pmids.py``).
  * The BM25 Lucene index (``--bm25-index``) — used as the abstract text
    source, since the official corpus dump is already indexed.

Output:
  * ``<out-dir>/index.faiss`` — ``IndexFlatIP`` over L2-normalised 768-d
    vectors (matches MedCPT's training-time similarity).
  * ``<out-dir>/pmid_lookup.parquet`` — row → PMID lookup; row ``i`` in
    the FAISS index is the PMID at row ``i`` of this parquet. The two
    files are written atomically (lookup last) so a partial run is
    detectable: lookup row count == index ntotal iff complete.
  * ``<out-dir>/build_progress.jsonl`` — one line per ``--batch-size``
    chunk; lets a crashed run resume by inspecting where it stopped.

Wall-clock budget: ~24 h on a 12-core CPU at batch 8. Resumable: pass
the same ``--out-dir`` to skip already-encoded shards.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl


DEFAULT_MODEL = "ncbi/MedCPT-Article-Encoder"
DEFAULT_BATCH = 8
DEFAULT_MAX_LEN = 512
DEFAULT_DIM = 768
SHARD_SIZE = 10_000   # write embeddings to disk in shards of this many vectors
PROGRESS_EVERY = 50   # log inner-loop throughput every N batches


@dataclass(slots=True)
class _Shard:
    idx: int            # 0-based shard index
    start: int          # first row of `pmid_lookup` this shard covers
    end: int            # one past the last row
    path: Path          # `shard_{idx:05d}.npy` under out_dir


def _shards_for(n_pmids: int, out_dir: Path) -> list[_Shard]:
    out: list[_Shard] = []
    idx = 0
    for start in range(0, n_pmids, SHARD_SIZE):
        end = min(start + SHARD_SIZE, n_pmids)
        out.append(_Shard(idx=idx, start=start, end=end,
                          path=out_dir / f"shard_{idx:05d}.npy"))
        idx += 1
    return out


def _encode_shard(
    pmids: list[str],
    *,
    doc_text_fn,
    tok,
    model,
    device,
    batch_size: int,
    max_len: int,
):
    """Encode one shard of PMIDs to a (len(pmids), dim) float32 array."""
    import numpy as np
    import torch

    import time

    out: list[np.ndarray] = []
    n_batches = (len(pmids) + batch_size - 1) // batch_size
    t0 = time.monotonic()
    for bi, i in enumerate(range(0, len(pmids), batch_size)):
        batch_pmids = pmids[i : i + batch_size]
        texts = [doc_text_fn(p) for p in batch_pmids]
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            output = model(**enc)
        # MedCPT-Article-Encoder uses [CLS]'s last hidden state, matching
        # the query encoder; L2-normalise so the FAISS IP index measures
        # cosine. Cast back to float32 for FAISS-IP compatibility (even if
        # the model ran in fp16).
        emb = output.last_hidden_state[:, 0, :].float().cpu().numpy().astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(norms, 1e-12, None)
        out.append(emb)
        if (bi + 1) % PROGRESS_EVERY == 0 or (bi + 1) == n_batches:
            elapsed = time.monotonic() - t0
            done = (bi + 1) * batch_size
            docs_per_sec = done / elapsed if elapsed > 0 else 0.0
            print(
                f"[build_dense]   batch {bi + 1}/{n_batches} "
                f"({done}/{len(pmids)} docs) "
                f"{docs_per_sec:.1f} docs/s elapsed={elapsed:.0f}s",
                flush=True,
            )
    return np.concatenate(out, axis=0) if out else np.zeros((0, DEFAULT_DIM), dtype="float32")


def _doc_text_factory(bm25_index_dir: Path):
    """Build a ``pmid -> text`` callable backed by the Lucene index."""
    from trec_biogen.retrieval.bm25 import BM25Index

    bm = BM25Index(bm25_index_dir)
    return bm, bm.doc_text


def build(
    *,
    pmids_parquet: Path,
    out_dir: Path,
    bm25_index_dir: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
    max_len: int = DEFAULT_MAX_LEN,
    device: str = "auto",
    fp16: bool = False,
    progress_log: Path | None = None,
) -> Path:
    """Encode every PMID in ``pmids_parquet`` and build the FAISS index."""
    import faiss
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    progress_log = progress_log or (out_dir / "build_progress.jsonl")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 and device == "cpu":
        # fp16 on CPU has no inference path in torch — silently downgrade.
        fp16 = False

    df = pl.read_parquet(pmids_parquet)
    if "pmid" not in df.columns:
        raise ValueError(f"{pmids_parquet} must have a 'pmid' column")
    pmids = [str(p) for p in df["pmid"].to_list()]
    n = len(pmids)
    print(
        f"[build_dense] encoding {n} pmids in shards of {SHARD_SIZE} "
        f"(device={device}, fp16={fp16}, batch={batch_size})",
        flush=True,
    )

    bm, doc_text_fn = _doc_text_factory(bm25_index_dir)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    if fp16:
        model = model.half()
    model = model.to(device)

    shards = _shards_for(n, out_dir)
    for shard in shards:
        if shard.path.exists() and shard.path.stat().st_size > 0:
            print(f"[build_dense] shard {shard.idx} already encoded — skip", flush=True)
            continue
        chunk = pmids[shard.start : shard.end]
        emb = _encode_shard(
            chunk,
            doc_text_fn=doc_text_fn,
            tok=tok, model=model, device=device,
            batch_size=batch_size, max_len=max_len,
        )
        np.save(shard.path, emb)
        with progress_log.open("a") as fh:
            fh.write(json.dumps({"shard": shard.idx, "n": emb.shape[0]}) + "\n")
        print(f"[build_dense] wrote shard {shard.idx} ({emb.shape[0]} vectors)", flush=True)

    bm.close()

    # Stitch shards into a single FAISS IndexFlatIP and write the index.
    dim = DEFAULT_DIM
    index = faiss.IndexFlatIP(dim)
    for shard in shards:
        emb = np.load(shard.path)
        if emb.shape[0] == 0:
            continue
        if emb.shape[1] != dim:
            raise ValueError(f"shard {shard.idx} has dim {emb.shape[1]}, expected {dim}")
        index.add(emb)
    out_index = out_dir / "index.faiss"
    faiss.write_index(index, str(out_index))

    # Lookup last — its existence + matching row count is the success marker.
    lookup_path = out_dir / "pmid_lookup.parquet"
    pl.DataFrame({"pmid": pmids}).write_parquet(lookup_path)
    print(
        f"[build_dense] done — {index.ntotal} vectors in {out_index}; "
        f"lookup at {lookup_path}",
        flush=True,
    )
    return out_index


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trec_biogen.retrieval.build_dense")
    p.add_argument("--pmids", type=Path, required=True,
                   help="Parquet with a single 'pmid' column (output of §9.1).")
    p.add_argument("--out-dir", type=Path, default=Path("data/indexes/medcpt_5m"))
    p.add_argument("--bm25-index", type=Path, default=Path("data/indexes/pubmed_bm25"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--device", default="auto",
                   help="cuda | cpu | auto (default: auto-detect)")
    p.add_argument("--fp16", action="store_true",
                   help="run the model in fp16 (CUDA only; CPU silently ignores)")
    args = p.parse_args(argv)
    build(
        pmids_parquet=args.pmids,
        out_dir=args.out_dir,
        bm25_index_dir=args.bm25_index,
        model_name=args.model,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=args.device,
        fp16=args.fp16,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
