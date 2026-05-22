"""Per-sentence selection of supports and contradictions (9.1, 9.2).

Rules (design D2, D9, plus official Task A track rule):
* Supports: top-3 by NLI entailment score above ``tau_sup``.
* Contradicts: top-3 by BM25 rank (lower rank = better) above ``tau_con``
  on the aggregated contradict score (the *simplicity paradox* finding).
* Exclude any PMID present in ``existing_supported_citations`` for that sentence
  (the track rule, also enforced by the starter-kit baseline).
* Global PMID dedup within a topic — a PMID may not appear in two sentences
  of the same answer. Lower-index sentence wins ties; promotion: when a
  conflict drops a candidate, the next-best from the loser's pool is promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from trec_biogen.io.topics import Topic


@dataclass(slots=True, frozen=True)
class SelectionConfig:
    tau_sup: float = 0.5      # min entailment_prob for support
    tau_con: float = 0.5      # min contradict_score for contradict
    cap: int = 3              # max PMIDs per (sentence, class)
    # Phase 2 §6: when False, do not exclude PMIDs listed in the input
    # answer's ``existing_supported_citations``. The official validator
    # still enforces the track rule downstream, so this only changes the
    # *internal* selection — useful for the ``phase2_allow_existing``
    # ablation variant that probes whether the rule is over-restrictive.
    exclude_existing: bool = True


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
        out.setdefault(key, []).append((str(row["candidate_pmid"]), float(row[score_col])))
    return out


def _existing_for(topics: list[Topic]) -> dict[tuple[str, int], set[str]]:
    """Index ``existing_supported_citations`` by (qa_id, sentence_id)."""
    out: dict[tuple[str, int], set[str]] = {}
    for t in topics:
        for sid, existing in enumerate(t.existing_per_sentence):
            out[(t.qa_id, sid)] = set(existing)
    return out


def select(
    *,
    nli_support: Path,
    nli_contradict: Path,
    topics: list[Topic] | None = None,
    config: SelectionConfig | None = None,
) -> dict[str, dict[int, dict[str, list[str]]]]:
    """Return ``{qa_id: {sentence_id: {"support": [...], "contradict": [...]}}}``.

    Applies thresholds, caps, ``existing_supported_citations`` exclusion,
    and global per-topic PMID dedup.
    """
    cfg = config or SelectionConfig()
    existing_idx = _existing_for(topics or [])

    sup_df = pl.read_parquet(nli_support)
    con_df = pl.read_parquet(nli_contradict)

    sup_pool = _candidates_per_sentence(
        sup_df, score_col="entailment_prob", threshold=cfg.tau_sup, sort_desc=True
    )
    # Contradict: filter by aggregated contradict_score, then order by BM25 rank
    # (lower rank = better) per the simplicity paradox finding (design D2).
    con_filtered = con_df.filter(pl.col("contradict_score") >= cfg.tau_con).sort("bm25_rank")
    con_pool: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for row in con_filtered.iter_rows(named=True):
        key = (row["qa_id"], int(row["sentence_id"]))
        con_pool.setdefault(key, []).append((str(row["candidate_pmid"]), float(row["bm25_rank"])))

    qa_ids = {q for (q, _) in sup_pool} | {q for (q, _) in con_pool}
    result: dict[str, dict[int, dict[str, list[str]]]] = {}

    for qa_id in qa_ids:
        cells = sorted(
            {s for (q, s) in sup_pool if q == qa_id}
            | {s for (q, s) in con_pool if q == qa_id}
        )
        used: set[str] = set()  # global dedup within topic
        topic_out: dict[int, dict[str, list[str]]] = {}
        for sid in cells:
            excluded = existing_idx.get((qa_id, sid), set()) if cfg.exclude_existing else set()

            chosen_sup: list[str] = []
            for pmid, _ in sup_pool.get((qa_id, sid), []):
                if len(chosen_sup) >= cfg.cap:
                    break
                if pmid in used or pmid in excluded:
                    continue
                chosen_sup.append(pmid)
                used.add(pmid)

            chosen_con: list[str] = []
            for pmid, _ in con_pool.get((qa_id, sid), []):
                if len(chosen_con) >= cfg.cap:
                    break
                if pmid in used or pmid in excluded:
                    continue
                chosen_con.append(pmid)
                used.add(pmid)

            topic_out[sid] = {"support": chosen_sup, "contradict": chosen_con}
        result[qa_id] = topic_out

    return result
