"""Reciprocal Rank Fusion (Phase 2 §9.4).

Fuses N independently-ranked candidate lists into a single ranking via::

    score(d) = sum over rankings r of  1 / (k + rank_r(d))

where ``k`` is a smoothing constant (Cormack et al., 2009 — default 60).
RRF is parameter-light, robust to score-scale differences across
rankers, and the de-facto baseline fusion method for hybrid sparse +
dense retrieval. The two inputs for the ``phase2_hybrid`` variant are
the BM25 ranked list and the MedCPT-Encoder FAISS ranked list per
(qa_id, sentence_id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from trec_biogen.retrieval.bm25 import Hit


@dataclass(slots=True, frozen=True)
class FusionEntry:
    """One row of an RRF result. ``score`` is the RRF score, not BM25 / cosine."""

    pmid: str
    rank: int
    score: float


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[Hit]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[FusionEntry]:
    """Combine multiple ranked lists into a single ranking via RRF.

    Parameters
    ----------
    rankings : iterable of iterable of :class:`Hit`
        Each input is a ranked list — order matters, but the per-hit
        ``rank`` attribute is used directly (so non-contiguous ranks
        are accepted).
    k : int, default 60
        Smoothing constant. The 60 default follows Cormack et al. (2009)
        and is the Pyserini / TREC standard.
    top_n : int | None
        If set, truncate the fused output to the top-``top_n`` entries.

    Returns
    -------
    list[FusionEntry]
        PMIDs ranked by descending RRF score. Ties are broken by PMID
        (lexicographically ascending) so the output is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for hit in ranking:
            scores[hit.pmid] = scores.get(hit.pmid, 0.0) + 1.0 / (k + hit.rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    if top_n is not None:
        fused = fused[:top_n]
    return [
        FusionEntry(pmid=pmid, rank=i + 1, score=score)
        for i, (pmid, score) in enumerate(fused)
    ]
