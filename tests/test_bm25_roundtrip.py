"""BM25 sentinel round-trip — task 4.4.

The Lucene index is multi-GB and built once overnight, so the test runs only
when the operator points it at a real index via ``BIOGEN_INDEX_DIR`` and a
sentinel PMID via ``BIOGEN_SENTINEL_PMID``. The fixture title is queried and
the sentinel PMID must appear in the top-10 hits.
"""

from __future__ import annotations

import os

import pytest

INDEX = os.environ.get("BIOGEN_INDEX_DIR")
PMID = os.environ.get("BIOGEN_SENTINEL_PMID")
TITLE = os.environ.get("BIOGEN_SENTINEL_TITLE")

requires_index = pytest.mark.skipif(
    not (INDEX and PMID and TITLE),
    reason="set BIOGEN_INDEX_DIR, BIOGEN_SENTINEL_PMID, BIOGEN_SENTINEL_TITLE",
)


@requires_index
def test_sentinel_title_retrieves_pmid() -> None:
    from trec_biogen.retrieval.bm25 import BM25Index

    bm = BM25Index(INDEX)
    try:
        hits = bm.search(TITLE, k=10)
        pmids = [h.pmid for h in hits]
        assert PMID in pmids, f"sentinel {PMID} not in top-10: {pmids}"
    finally:
        bm.close()


@requires_index
def test_doc_count_in_expected_range() -> None:
    from trec_biogen.retrieval.bm25 import BM25Index

    bm = BM25Index(INDEX)
    try:
        n = bm.doc_count()
        assert 26_000_000 <= n <= 27_500_000, f"unexpected doc count: {n}"
    finally:
        bm.close()
