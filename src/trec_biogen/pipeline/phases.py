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

from tqdm.auto import tqdm

from trec_biogen.io.topics import Topic
from trec_biogen.pipeline.sentences import split_sentences
from trec_biogen.retrieval.bm25 import BM25Index, Hit
from trec_biogen.retrieval.dense import DenseIndex
from trec_biogen.retrieval.llm_prf import (
    LLMRelevanceFilter,
    build_expanded_query,
    manual_rm3_terms,
)
from trec_biogen.retrieval.rrf import reciprocal_rank_fusion


def _query_for(topic: Topic, sentence: str) -> str:
    """Concatenate question + sentence — design uses (question + sentence) for both paths."""
    return f"{topic.question} {sentence}".strip()


def retrieve(
    topics: Iterable[Topic],
    bm25: BM25Index,
    *,
    k: int,
    out_path: Path,
    rm3: bool = False,
) -> Path:
    """Run BM25 at depth ``k`` for every (topic, answer-sentence) pair.

    Uses the pre-segmented ``topic.sentences`` (one row per official ``answer[i]``).
    ``rm3=True`` (Phase 2 §8) enables Pyserini's RM3 query expansion on the
    underlying searcher for the duration of this call.
    """
    rows: list[dict] = []
    for topic in topics:
        for sid, sent in enumerate(topic.sentences):
            query = _query_for(topic, sent)
            hits: list[Hit] = bm25.search(query, k=k, rm3=rm3)
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


def retrieve_hybrid(
    topics: Iterable[Topic],
    bm25: BM25Index,
    dense: DenseIndex,
    *,
    k: int,
    out_path: Path,
    rrf_k: int = 60,
    leg_k: int | None = None,
) -> Path:
    """Hybrid BM25 + Dense retrieval, fused via Reciprocal Rank Fusion.

    Per (topic, answer-sentence): run BM25 at depth ``leg_k`` (default
    ``k``) AND dense at depth ``leg_k``, fuse via RRF with smoothing
    ``rrf_k``, keep the top ``k`` fused candidates.

    Output schema matches :func:`retrieve` — ``candidate_pmid``,
    ``rank`` (RRF rank), ``bm25_score`` (RRF score; the column name is
    kept for downstream compatibility but the value is the fused score,
    not raw BM25). Phase 2 §9.6 dispatches to this function when
    ``cfg.retrieval.flavour == "hybrid_rrf"``.
    """
    leg_k = leg_k or k
    rows: list[dict] = []
    for topic in topics:
        for sid, sent in enumerate(topic.sentences):
            query = _query_for(topic, sent)
            bm_hits = bm25.search(query, k=leg_k)
            dense_hits_raw = dense.search(query, k=leg_k)
            # DenseHit and Hit are different dataclasses; RRF only reads
            # .pmid and .rank, which both expose.
            fused = reciprocal_rank_fusion(
                [bm_hits, dense_hits_raw], k=rrf_k, top_n=k,
            )
            for entry in fused:
                rows.append(
                    {
                        "qa_id": topic.qa_id,
                        "sentence_id": sid,
                        "sentence_text": sent,
                        "query": query,
                        "candidate_pmid": entry.pmid,
                        "rank": entry.rank,
                        "bm25_score": entry.score,
                    }
                )
    df = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return out_path


def retrieve_llm_filtered_rm3(
    topics: Iterable[Topic],
    bm25: BM25Index,
    llm_filter: LLMRelevanceFilter,
    *,
    k: int,
    out_path: Path,
    initial_k: int = 30,
    fb_terms: int = 10,
    _logger=None,
) -> Path:
    """BM25 + LLM-filtered RM3 retrieval (Phase 2 §12.7).

    Per (topic, answer-sentence) cell:

    1. BM25 top-``initial_k`` (default 30).
    2. LLM binary filter `relevant?` per candidate.
    3. Manual RM1-style expansion-term extraction over LLM-accepted
       docs (skipped if zero accepted → falls back to plain BM25
       top-``k`` for the cell).
    4. BM25 with the expanded query at depth ``k``.

    Output schema matches :func:`retrieve` (same parquet columns) so
    the downstream rerank / NLI / selection stages are unchanged.
    ``query`` column stores the *expanded* query — useful for
    debugging which expansion terms each cell actually used.
    """
    from pyserini.index.lucene import LuceneIndexReader

    reader = LuceneIndexReader(str(bm25.index_dir))
    rows: list[dict] = []
    n_filter_calls = 0
    n_cells_skipped = 0
    n_cells_expanded = 0
    for topic in topics:
        for sid, sent in enumerate(topic.sentences):
            query = _query_for(topic, sent)
            initial = bm25.search(query, k=initial_k)
            if not initial:
                continue
            # Fetch abstract text once per candidate.
            candidates = [(h.pmid, bm25.doc_text(h.pmid)) for h in initial]
            decisions = llm_filter.filter_many(sent, candidates)
            n_filter_calls += len(decisions)
            accepted = [d.pmid for d in decisions if d.relevant]

            if accepted:
                expansion = manual_rm3_terms(
                    reader, accepted, fb_terms=fb_terms,
                    exclude=query.lower().split(),
                )
                expanded_query = build_expanded_query(query, expansion)
                final_hits = bm25.search(expanded_query, k=k)
                n_cells_expanded += 1
            else:
                expanded_query = query
                final_hits = bm25.search(query, k=k)
                n_cells_skipped += 1
            for h in final_hits:
                rows.append(
                    {
                        "qa_id": topic.qa_id,
                        "sentence_id": sid,
                        "sentence_text": sent,
                        "query": expanded_query,
                        "candidate_pmid": h.pmid,
                        "rank": h.rank,
                        "bm25_score": h.score,
                    }
                )
        if _logger is not None:
            _logger.info(
                f"[llm_prf] qa_id={topic.qa_id}: filter calls so far "
                f"= {n_filter_calls}; cells expanded {n_cells_expanded}, "
                f"fallback {n_cells_skipped}; "
                f"cost so far ${llm_filter.stats['cost_usd']:.4f}"
            )

    df = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return out_path


def retrieve_llm_rewrite(
    topics: Iterable[Topic],
    bm25: BM25Index,
    rewriter: LLMQueryRewriter,
    *,
    k: int,
    out_path: Path,
    rrf_k: int = 60,
    include_original: bool = True,
    _logger=None,
) -> Path:
    """BM25 retrieval over LLM-generated query variants, fused via RRF
    (Phase 2 §12.10).

    Per (topic, answer-sentence): ask the LLM rewriter for ``n_variants``
    short biomedical queries; optionally include the original
    ``question + sentence`` query too (default ``True``); run BM25 over
    each variant at depth ``k``; fuse via RRF with smoothing ``rrf_k``;
    keep the top-``k`` fused candidates.

    Output schema matches :func:`retrieve`. The ``query`` column stores
    the concatenation of the original + variant strings (debug-friendly)
    so the parquet records what was actually fed to BM25.
    """
    # Build the list of (question, sentence) pairs we need rewrites for.
    items: list[tuple[str, str, str, int]] = []   # (question, sentence, qa_id, sid)
    for topic in topics:
        for sid, sent in enumerate(topic.sentences):
            items.append((topic.question, sent, topic.qa_id, sid))
    if _logger is not None:
        _logger.info(f"[llm_rewrite] requesting rewrites for {len(items)} cells")
    records = rewriter.rewrite_many([(q, s) for q, s, _, _ in items])
    if _logger is not None:
        _logger.info(
            f"[llm_rewrite] rewrite cost: ${rewriter.stats['cost_usd']:.4f} "
            f"over {rewriter.stats['calls']} calls"
        )

    rows: list[dict] = []
    for (question, sent, qa_id, sid), rec in zip(items, records):
        original_query = _query_for_topic_question_sentence(question, sent)
        rankings: list[list[Hit]] = []
        queries_used: list[str] = []
        if include_original:
            rankings.append(bm25.search(original_query, k=k))
            queries_used.append(original_query)
        for q in rec.queries:
            rankings.append(bm25.search(q, k=k))
            queries_used.append(q)
        if not rankings:
            continue
        fused = reciprocal_rank_fusion(rankings, k=rrf_k, top_n=k)
        # Concatenate the variants used for downstream debuggability.
        joined_query = " || ".join(queries_used)
        for entry in fused:
            rows.append(
                {
                    "qa_id": qa_id,
                    "sentence_id": sid,
                    "sentence_text": sent,
                    "query": joined_query,
                    "candidate_pmid": entry.pmid,
                    "rank": entry.rank,
                    "bm25_score": entry.score,
                }
            )
    df = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return out_path


def _query_for_topic_question_sentence(question: str, sentence: str) -> str:
    """Same convention as ``_query_for(topic, sentence)`` but takes the strings
    directly. Lets ``retrieve_llm_rewrite`` reconstruct the original query
    without holding the ``Topic`` object around."""
    return f"{question.strip()} {sentence.strip()}".strip()


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
    # Progress bar advances per PMID (the dominant cost = Lucene fetch + scispaCy parse).
    seg_cache: dict[str, list[str]] = {}
    for pmid in tqdm(unique_pmids, desc="segment_abstracts", unit="pmid"):
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
