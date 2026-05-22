"""Unit tests for Reciprocal Rank Fusion (Phase 2 §9.10)."""

from __future__ import annotations

import pytest

from trec_biogen.retrieval.bm25 import Hit
from trec_biogen.retrieval.rrf import reciprocal_rank_fusion


def _hits(ranked_pmids: list[str]) -> list[Hit]:
    return [Hit(pmid=p, rank=i + 1, score=100.0 - i) for i, p in enumerate(ranked_pmids)]


def test_single_ranking_passthrough() -> None:
    """One ranking → fused order matches input, RRF score = 1/(k+rank)."""
    fused = reciprocal_rank_fusion([_hits(["A", "B", "C"])], k=60)
    assert [e.pmid for e in fused] == ["A", "B", "C"]
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)
    assert fused[2].score == pytest.approx(1 / 63)
    # Output ranks are 1-based dense.
    assert [e.rank for e in fused] == [1, 2, 3]


def test_overlap_boosts_shared_pmids() -> None:
    """A PMID present in both rankings beats one present in just one."""
    fused = reciprocal_rank_fusion(
        [
            _hits(["A", "B", "C"]),
            _hits(["A", "X", "Y"]),
        ],
        k=60,
    )
    # A appears at rank 1 in both → score = 2/61
    # B appears at rank 2 in one → score = 1/62
    # X at rank 2 in one          → score = 1/62, but tied with B; sorted by pmid asc
    a = next(e for e in fused if e.pmid == "A")
    b = next(e for e in fused if e.pmid == "B")
    x = next(e for e in fused if e.pmid == "X")
    assert a.score == pytest.approx(2 / 61)
    assert b.score == pytest.approx(1 / 62)
    assert x.score == pytest.approx(1 / 62)
    assert a.rank < b.rank
    # Tie-break is deterministic (pmid ascending).
    assert fused.index(b) < fused.index(x)


def test_disjoint_rankings_concatenate_by_score() -> None:
    fused = reciprocal_rank_fusion(
        [
            _hits(["A", "B"]),
            _hits(["C", "D"]),
        ],
        k=60,
    )
    # All four appear with rank ∈ {1, 2} from a single ranking.
    pmids = sorted(e.pmid for e in fused)
    assert pmids == ["A", "B", "C", "D"]
    rank1_pmids = {e.pmid for e in fused if e.score == pytest.approx(1 / 61)}
    rank2_pmids = {e.pmid for e in fused if e.score == pytest.approx(1 / 62)}
    assert rank1_pmids == {"A", "C"}
    assert rank2_pmids == {"B", "D"}


def test_k_parameter_changes_score_but_not_order_when_only_one_ranking() -> None:
    a = reciprocal_rank_fusion([_hits(["A", "B"])], k=60)
    b = reciprocal_rank_fusion([_hits(["A", "B"])], k=30)
    assert [e.pmid for e in a] == [e.pmid for e in b]
    assert a[0].score < b[0].score   # smaller k → larger score


def test_top_n_truncates() -> None:
    fused = reciprocal_rank_fusion(
        [_hits(["A", "B", "C", "D", "E"])], k=60, top_n=3,
    )
    assert [e.pmid for e in fused] == ["A", "B", "C"]
    assert [e.rank for e in fused] == [1, 2, 3]


def test_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_deterministic_tie_break() -> None:
    """When two PMIDs have identical RRF scores, they sort by PMID ascending."""
    fused = reciprocal_rank_fusion(
        [_hits(["Z", "M", "A"])], k=60,
    )
    # No ties here, but reverse alphabetical input — order driven by rank, not name.
    assert [e.pmid for e in fused] == ["Z", "M", "A"]

    # Force a tie: two PMIDs at rank 1 in different rankings.
    fused = reciprocal_rank_fusion(
        [_hits(["Z"]), _hits(["A"])], k=60,
    )
    # Both have score 1/61; pmid-asc tie break → A before Z.
    assert [e.pmid for e in fused] == ["A", "Z"]
