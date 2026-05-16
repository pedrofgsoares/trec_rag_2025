"""Phase functions invoked by ``run_task_a``. Each writes a Parquet file under
``runs/<id>/`` and can be re-run in isolation against the upstream output
(design D6).

Tasks: 7.1 (support retrieval), 8.1 (contradict retrieval), 8.2 (abstract
sentence segmentation), 8.5 (max-pool contradiction aggregation).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import polars as pl

from trec_biogen.io.topics import Topic
from trec_biogen.pipeline.sentences import split_sentences
from trec_biogen.retrieval.bm25 import BM25Index, Hit


def _query_for(topic: Topic, sentence: str) -> str:
    """Concatenate question + sentence — design uses (question + sentence) for both paths."""
    return f"{topic.question} {sentence}".strip()


def retrieve(
    topics: Iterable[Topic],
    bm25: BM25Index,
    *,
    k: int,
    out_path: Path,
) -> Path:
    """Run BM25 at depth ``k`` for every (topic, answer-sentence) pair.

    Uses the pre-segmented ``topic.sentences`` (one row per official ``answer[i]``).
    """
    rows: list[dict] = []
    for topic in topics:
        for sid, sent in enumerate(topic.sentences):
            query = _query_for(topic, sent)
            hits: list[Hit] = bm25.search(query, k=k)
            for h in hits:
                rows.append(
                    {
                        "qa_id": topic.qa_id,
                        "sentence_id": sid,
                        "sentence_text": sent,
                        "query": query,
                        "candidate_pmid": h.pmid,
                        "rank": h.rank,
                        "bm25_score": h.score,
                    }
                )
    df = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return out_path


def segment_abstracts(
    retrieval_parquet: Path,
    bm25: BM25Index,
    *,
    out_path: Path,
) -> Path:
    """Task 8.2 — one row per (candidate_pmid, abstract_sentence_idx, ...).

    Reads the deep retrieval Parquet (k=1000), fetches each candidate's stored
    contents from the Lucene index, segments via scispaCy, and writes a long
    table. Sentences with empty text are dropped.
    """
    df = pl.read_parquet(retrieval_parquet)
    unique_pmids = df["candidate_pmid"].unique().to_list()

    # Cache pmid -> [sentence] to avoid re-segmenting popular docs.
    seg_cache: dict[str, list[str]] = {}
    for pmid in unique_pmids:
        text = bm25.doc_text(pmid)
        seg_cache[pmid] = split_sentences(text) if text else []

    rows: list[dict] = []
    for row in df.iter_rows(named=True):
        for idx, sent in enumerate(seg_cache.get(row["candidate_pmid"], [])):
            rows.append(
                {
                    "qa_id": row["qa_id"],
                    "sentence_id": row["sentence_id"],
                    "candidate_pmid": row["candidate_pmid"],
                    "abstract_sentence_idx": idx,
                    "abstract_sentence_text": sent,
                    "bm25_rank": row["rank"],
                    "bm25_score": row["bm25_score"],
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out_path)
    return out_path


def aggregate_contradict(
    nli_pairs_parquet: Path,
    *,
    score_col: str = "contradiction_prob",
    out_path: Path,
) -> Path:
    """Task 8.5 — max-pool sentence-pair contradiction scores into per-document scores."""
    df = pl.read_parquet(nli_pairs_parquet)
    agg = (
        df.group_by(["qa_id", "sentence_id", "candidate_pmid"])
        .agg(
            pl.col(score_col).max().alias("contradict_score"),
            pl.col("bm25_rank").min().alias("bm25_rank"),
            pl.col("bm25_score").max().alias("bm25_score"),
        )
        .sort(["qa_id", "sentence_id", "bm25_rank"])
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.write_parquet(out_path)
    return out_path
