"""Per-sentence selection of supports and contradictions (9.1, 9.2).

Rules (design D2, D9):
* Supports: top-3 by NLI entailment score above ``tau_sup``.
* Contradicts: top-3 by BM25 rank (lower rank = better) above ``tau_con``
  on the aggregated contradict score (the *simplicity paradox* finding).
* Global PMID dedup within a topic — a PMID may not appear in two sentences
  of the same answer. Lower-index sentence wins ties; promotion: when a
  conflict drops a candidate, the next-best from the loser's pool is promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(slots=True, frozen=True)
class SelectionConfig:
    tau_sup: float = 0.5      # min entailment_prob for support
    tau_con: float = 0.5      # min contradict_score for contradict
    cap: int = 3              # max PMIDs per (sentence, class)


def _candidates_per_sentence(
    df: pl.DataFrame,
    *,
    score_col: str,
    threshold: float,
    sort_desc: bool,
) -> dict[tuple[str, int], list[tuple[str, float]]]:
    """Filter+sort into ordered ``[(pmid, score), ...]`` per (qa_id, sentence_id)."""
    out: dict[tuple[str, int], list[tuple[str, float]]] = {}
    filt = df.filter(pl.col(score_col) >= threshold)
    sorted_df = filt.sort(score_col, descending=sort_desc)
    for row in sorted_df.iter_rows(named=True):
        key = (row["qa_id"], int(row["sentence_id"]))
        out.setdefault(key, []).append((row["candidate_pmid"], float(row[score_col])))
    return out


def select(
    *,
    nli_support: Path,
    nli_contradict: Path,
    config: SelectionConfig | None = None,
) -> dict[str, dict[int, dict[str, list[str]]]]:
    """Return ``{qa_id: {sentence_id: {"support": [...], "contradict": [...]}}}``.

    Applies thresholds, caps, and global per-topic PMID dedup.
    """
    cfg = config or SelectionConfig()

    sup_df = pl.read_parquet(nli_support)
    # For contradicts, sort by BM25 rank ascending (better rank first); the
    # aggregated parquet keeps ``bm25_rank`` from upstream.
    con_df = pl.read_parquet(nli_contradict)

    sup_pool = _candidates_per_sentence(
        sup_df, score_col="entailment_prob", threshold=cfg.tau_sup, sort_desc=True
    )
    # Contradict path: filter by aggregated contradict_score, *order by BM25 rank ascending*
    # (lower rank = better) per the simplicity paradox finding (design D2).
    con_filtered = con_df.filter(pl.col("contradict_score") >= cfg.tau_con).sort("bm25_rank")
    con_pool: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for row in con_filtered.iter_rows(named=True):
        key = (row["qa_id"], int(row["sentence_id"]))
        con_pool.setdefault(key, []).append((row["candidate_pmid"], float(row["bm25_rank"])))

    qa_ids = {q for (q, _) in sup_pool} | {q for (q, _) in con_pool}
    result: dict[str, dict[int, dict[str, list[str]]]] = {}

    for qa_id in qa_ids:
        # Collect all (qa_id, sentence_id) cells for this topic.
        cells = sorted(
            {s for (q, s) in sup_pool if q == qa_id}
            | {s for (q, s) in con_pool if q == qa_id}
        )
        used: set[str] = set()  # global dedup within topic
        topic_out: dict[int, dict[str, list[str]]] = {}
        # Process supports first then contradicts in sentence order; the
        # winner of a PMID conflict is the earlier sentence that selects it.
        # Both classes draw from disjoint pools by construction (different
        # retrieval depths), but we still dedup to satisfy the rule strictly.
        for sid in cells:
            chosen_sup: list[str] = []
            for pmid, _ in sup_pool.get((qa_id, sid), []):
                if len(chosen_sup) >= cfg.cap:
                    break
                if pmid in used:
                    continue
                chosen_sup.append(pmid)
                used.add(pmid)

            chosen_con: list[str] = []
            for pmid, _ in con_pool.get((qa_id, sid), []):
                if len(chosen_con) >= cfg.cap:
                    break
                if pmid in used:
                    continue
                chosen_con.append(pmid)
                used.add(pmid)

            topic_out[sid] = {"support": chosen_sup, "contradict": chosen_con}
        result[qa_id] = topic_out

    return result
