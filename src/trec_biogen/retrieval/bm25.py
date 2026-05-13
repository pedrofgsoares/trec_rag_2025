"""Pyserini BM25 search wrapper.

A single ``LuceneSearcher`` is held open by ``BM25Index`` and re-used across
both k=100 (support path) and k=1000 (contradiction path) queries — see
design D1 / D2. The class is intentionally minimal: configure once, search
many times.

Task: 4.3, 4.5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing pyserini at module-import time
    from pyserini.search.lucene import LuceneSearcher


@dataclass(slots=True, frozen=True)
class Hit:
    pmid: str
    rank: int
    score: float


class BM25Index:
    """Open a Pyserini Lucene index and serve BM25 ranked lists.

    Parameters
    ----------
    index_dir : path to a directory built by ``scripts/build_indexes.sh``.
    k1, b : BM25 parameters; defaults match Anserini's PubMed presets.
    """

    def __init__(
        self,
        index_dir: Path | str,
        *,
        k1: float = 0.9,
        b: float = 0.4,
    ) -> None:
        from pyserini.search.lucene import LuceneSearcher  # local import: heavy JVM

        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(f"index not found: {self.index_dir}")
        self._searcher: LuceneSearcher = LuceneSearcher(str(self.index_dir))
        self._searcher.set_bm25(k1=k1, b=b)

    def search(self, query: str, k: int) -> list[Hit]:
        """Run BM25, return up to ``k`` hits sorted by descending score."""
        raw = self._searcher.search(query, k=k)
        return [Hit(pmid=h.docid, rank=i + 1, score=h.score) for i, h in enumerate(raw)]

    def doc_count(self) -> int:
        return self._searcher.num_docs

    def doc_text(self, pmid: str) -> str:
        """Return the stored ``contents`` (title + abstract) for a PMID, or ''.

        Requires the index to have been built with ``--storeRaw``.
        """
        doc = self._searcher.doc(pmid)
        if doc is None:
            return ""
        raw = doc.raw()
        if not raw:
            return doc.contents() or ""
        try:
            import orjson

            return orjson.loads(raw).get("contents", "") or ""
        except Exception:
            return ""

    def close(self) -> None:
        # LuceneSearcher has no explicit close; releasing the reference is enough.
        self._searcher = None  # type: ignore[assignment]


def verify_index(index_dir: Path | str, *, min_docs: int = 26_000_000) -> None:
    """Preflight: open the index and assert it looks like the BioGen corpus.

    Raises ``RuntimeError`` if the index is missing or unexpectedly small.
    Used by the pipeline entry point before any GPU work starts.
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise RuntimeError(f"BM25 index missing: {index_dir} (build via scripts/build_indexes.sh)")
    bm = BM25Index(index_dir)
    n = bm.doc_count()
    bm.close()
    if n < min_docs:
        raise RuntimeError(
            f"BM25 index has {n} docs, expected ≥ {min_docs}. Did indexing complete?"
        )
