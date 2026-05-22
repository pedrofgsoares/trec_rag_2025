"""Tests for the §2.17 expand-pool helper.

Covers:
* ``top_k_triples_from_retrieval_parquets`` filters by rank, dedupes
  across paths, and honours ``exclude``.
* ``human_qrels_keys`` collects every human qrels triple.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import polars as pl

from trec_biogen.judge.rejudge import (
    human_qrels_keys,
    top_k_triples_from_retrieval_parquets,
)


def _write_retrieval_parquet(path: Path, rows: list[dict]) -> Path:
    pl.DataFrame(rows).write_parquet(path)
    return path


def _retrieval_rows(qa_id: str, sid: int, pmids_by_rank: list[str]) -> list[dict]:
    return [
        {
            "qa_id": qa_id,
            "sentence_id": sid,
            "sentence_text": "s",
            "query": "q",
            "candidate_pmid": pmid,
            "rank": rank,
            "bm25_score": 100.0 - rank,
        }
        for rank, pmid in enumerate(pmids_by_rank, start=1)
    ]


def test_top_k_filters_by_rank(tmp_path: Path) -> None:
    p = _write_retrieval_parquet(
        tmp_path / "sup.parquet",
        _retrieval_rows("Q1", 0, [f"P{i}" for i in range(1, 11)]),
    )
    out = top_k_triples_from_retrieval_parquets([p], top_k=3)
    pmids = sorted(t[2] for t in out)
    assert pmids == ["P1", "P2", "P3"]


def test_top_k_dedupes_across_paths(tmp_path: Path) -> None:
    sup = _write_retrieval_parquet(
        tmp_path / "sup.parquet",
        _retrieval_rows("Q1", 0, ["A", "B", "C"]),
    )
    con = _write_retrieval_parquet(
        tmp_path / "con.parquet",
        _retrieval_rows("Q1", 0, ["A", "D", "E"]),  # A overlaps with sup
    )
    out = top_k_triples_from_retrieval_parquets([sup, con], top_k=3)
    pmids = sorted(t[2] for t in out)
    assert pmids == ["A", "B", "C", "D", "E"]  # 5 unique, not 6


def test_top_k_excludes_given_keys(tmp_path: Path) -> None:
    p = _write_retrieval_parquet(
        tmp_path / "sup.parquet",
        _retrieval_rows("Q1", 0, ["A", "B", "C", "D"]),
    )
    excluded = {("Q1", 0, "A"), ("Q1", 0, "B")}
    out = top_k_triples_from_retrieval_parquets([p], top_k=4, exclude=excluded)
    pmids = sorted(t[2] for t in out)
    assert pmids == ["C", "D"]


def test_top_k_multi_cell(tmp_path: Path) -> None:
    p = _write_retrieval_parquet(
        tmp_path / "sup.parquet",
        _retrieval_rows("Q1", 0, ["A", "B"]) + _retrieval_rows("Q2", 1, ["C", "D"]),
    )
    out = top_k_triples_from_retrieval_parquets([p], top_k=10)
    assert ("Q1", 0, "A") in out
    assert ("Q1", 0, "B") in out
    assert ("Q2", 1, "C") in out
    assert ("Q2", 1, "D") in out
    assert len(out) == 4


def test_human_qrels_keys_collects_everything(tmp_path: Path) -> None:
    qrels = tmp_path / "qrels.jsonl"
    rows = [
        {"qa_id": "Q1", "sentence_id": 0, "pmid": "P1",
         "class": "support", "relevance": 1},
        {"qa_id": "Q1", "sentence_id": 0, "pmid": "P2",
         "class": "contradict", "relevance": 1},
        {"qa_id": "Q2", "sentence_id": 1, "pmid": "P3",
         "class": "partial_support", "relevance": 1},
    ]
    qrels.write_text("\n".join(orjson.dumps(r).decode() for r in rows) + "\n")
    keys = human_qrels_keys(qrels)
    assert keys == {("Q1", 0, "P1"), ("Q1", 0, "P2"), ("Q2", 1, "P3")}


def test_cmd_rejudge_still_works_after_refactor(tmp_path: Path, monkeypatch) -> None:
    """Smoke test: the refactor that extracted ``_judge_triples_and_emit``
    did not break the rejudge subcommand wiring. We don't run a real
    judge — instead we verify the parser accepts the rejudge args.
    """
    from trec_biogen.judge.rejudge import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "rejudge",
        "--backend", "openai-mini",
        "--submission", "/tmp/fake.json",
        "--qrels", "/tmp/fake.jsonl",
        "--topics", "/tmp/fake_topics.json",
        "--index", "/tmp/fake_index",
        "--out", "/tmp/fake_expanded.jsonl",
    ])
    assert args.cmd == "rejudge"
    assert args.backend == "openai-mini"


def test_cmd_expand_pool_parser_accepts_required_args() -> None:
    from trec_biogen.judge.rejudge import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "expand-pool",
        "--backend", "openai-mini",
        "--prompt", "cot",
        "--retrieval-support", "runs/x/retrieval_support.parquet",
        "--retrieval-contradict", "runs/x/retrieval_contradict.parquet",
        "--top-k", "30",
        "--qrels", "data/qrels/biogen2025_taskA_qrels.jsonl",
        "--topics", "data/topics/biogen2025_taskA_input.json",
        "--index", "data/indexes/pubmed_bm25",
        "--out", "data/qrels/biogen2025_taskA_qrels_expanded.jsonl",
        "--cost-cap", "10",
        "--max-concurrent", "8",
    ])
    assert args.cmd == "expand-pool"
    assert args.top_k == 30
    assert args.cost_cap == 10.0
    assert args.max_concurrent == 8
