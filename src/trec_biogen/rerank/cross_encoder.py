"""MedCPT cross-encoder reranker for the support path (D5, task 7.2).

Loads ``ncbi/MedCPT-Cross-Encoder`` at batch 8, scores every (sentence,
title+abstract) pair from the support retrieval Parquet, and writes the
top-30 per (qa_id, sentence_id) to ``rerank_support.parquet``.

The model is loaded once on entry and unloaded by the orchestrator after
this phase completes (sequential loading, design D6).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trec_biogen.pipeline.model_utils import device
from trec_biogen.retrieval.bm25 import BM25Index

DEFAULT_MODEL = "ncbi/MedCPT-Cross-Encoder"
DEFAULT_BATCH = 8
DEFAULT_TOP_K = 30
MAX_LEN = 512


def rerank_support(
    retrieval_parquet: Path,
    bm25: BM25Index,
    *,
    out_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
    top_k: int = DEFAULT_TOP_K,
) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()

    df = pl.read_parquet(retrieval_parquet)
    # Materialise the doc text once per unique pmid.
    unique_pmids = df["candidate_pmid"].unique().to_list()
    doc_text = {pmid: bm25.doc_text(pmid) for pmid in unique_pmids}

    rows = df.to_dicts()
    pairs = [(r["sentence_text"], doc_text.get(r["candidate_pmid"], "")) for r in rows]

    scores: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = tok.batch_encode_plus(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(dev)
            logits = model(**enc).logits.squeeze(-1)
            scores.extend(logits.cpu().tolist())

    enriched = (
        pl.DataFrame(rows)
        .with_columns(pl.Series("ce_score", scores))
        .sort(["qa_id", "sentence_id", "ce_score"], descending=[False, False, True])
        .group_by(["qa_id", "sentence_id"], maintain_order=True)
        .head(top_k)
        .with_columns(
            pl.cum_count("ce_score")
            .over(["qa_id", "sentence_id"])
            .alias("rank_after_rerank")
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.write_parquet(out_path)
    return out_path
