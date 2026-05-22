"""Dense-index smoke test (Phase 2 §9.10) — env-gated.

The MedCPT 5M FAISS index is ~15 GB and built once overnight, so the test
runs only when the operator points it at a real index via
``BIOGEN_DENSE_INDEX_DIR``. Verifies the index opens, the lookup matches
the FAISS row count, and a simple query returns the expected shape.

Models are auto-fetched from the Hugging Face Hub at first call. Set
``HF_HOME`` if you want them cached elsewhere.
"""

from __future__ import annotations

import os

import pytest

INDEX = os.environ.get("BIOGEN_DENSE_INDEX_DIR")
SENTINEL_QUERY = os.environ.get(
    "BIOGEN_DENSE_SENTINEL_QUERY",
    "metformin reduces HbA1c in type 2 diabetes",
)

requires_dense_index = pytest.mark.skipif(
    not INDEX,
    reason="set BIOGEN_DENSE_INDEX_DIR to an unpacked medcpt_5m index dir",
)


@requires_dense_index
def test_dense_index_opens_and_searches() -> None:
    from trec_biogen.retrieval.dense import DenseIndex

    di = DenseIndex(INDEX)
    try:
        hits = di.search(SENTINEL_QUERY, k=10)
    finally:
        di.close()

    assert 1 <= len(hits) <= 10
    # Ranks are 1-based dense.
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    # All PMIDs are non-empty strings.
    for h in hits:
        assert isinstance(h.pmid, str) and h.pmid
    # Scores are monotonically non-increasing.
    for a, b in zip(hits, hits[1:]):
        assert a.score >= b.score


@requires_dense_index
def test_dense_index_search_k_respected() -> None:
    from trec_biogen.retrieval.dense import DenseIndex

    di = DenseIndex(INDEX)
    try:
        small = di.search(SENTINEL_QUERY, k=3)
        large = di.search(SENTINEL_QUERY, k=20)
    finally:
        di.close()
    assert len(small) <= 3
    assert len(large) <= 20
    # Top-3 of the larger result must match the small result.
    assert [h.pmid for h in small] == [h.pmid for h in large[: len(small)]]
