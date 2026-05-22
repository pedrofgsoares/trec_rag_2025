"""Phase 2.5 §4 — per-topic F1 aggregation.

The cell-level macro that `eval/metrics.py` exposes is the published
methodology and remains the headline number. This module adds a
*topic-level* aggregation alongside it: for every `qa_id` present in a
run's submission, compute the mean per-cell F1 across that topic's
cells, per (class, setting). Used by the qualitative analysis to pick
3 representative topics and to surface where the pipeline gains, ties,
or loses against an anchor run.

The aggregation reads from cached `task_a_output.json`; no compute is
required beyond a re-scoring pass against the chosen qrels pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from trec_biogen.eval.metrics import (
    DEFAULT_QRELS_PATHS,
    _CLASSES,
    _SETTINGS,
    _iter_submission_cells,
    _prf,
    PRF,
)
from trec_biogen.io.qrels import QrelsIndex, Source, load_qrels


Pool = Literal["official", "expanded", "intersection"]


@dataclass(slots=True)
class TopicMetrics:
    """Per-topic macro derived from cell-level PRFs.

    `n_cells` counts cells included in the mean (i.e. cells that survived
    the `unjudged_as_zero` rule). For Phase 2.5 sentence-level reporting
    we use `unjudged_as_zero=True` (matches the published BioGEN 2025
    protocol and the rest of the pipeline).
    """

    P: float
    R: float
    F1: float
    n_cells: int

    def as_dict(self) -> dict[str, float | int]:
        return {"P": self.P, "R": self.R, "F1": self.F1, "n_cells": self.n_cells}


@dataclass(slots=True)
class PerTopicReport:
    """`topics[qa_id][setting][class]` -> TopicMetrics."""

    pool: str
    qrels_path: Path
    topics: dict[str, dict[str, dict[str, TopicMetrics]]] = field(default_factory=dict)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def per_topic_f1(
    run_dir: Path,
    *,
    pool: Pool = "intersection",
    qrels_path: Path | None = None,
    source: Source = "any",
    unjudged_as_zero: bool = True,
) -> PerTopicReport:
    """Compute per-topic macro F1 for one run.

    Args:
      run_dir: a completed run with ``task_a_output.json`` inside.
      pool: which canonical qrels pool to score against. Default
        ``"intersection"`` so the qualitative analysis defaults to the
        conservative pool. ``qrels_path`` overrides this.
      qrels_path: explicit qrels JSONL path. Overrides ``pool``.
      source: standard source filter (human / llm / any).
      unjudged_as_zero: matches the cell-level macro convention; cells
        with empty positives but non-empty predictions contribute F1=0.
    """
    submission = run_dir / "task_a_output.json"
    if not submission.exists():
        raise FileNotFoundError(
            f"{run_dir} has no task_a_output.json — cannot compute per-topic F1"
        )

    chosen_path = qrels_path if qrels_path is not None else DEFAULT_QRELS_PATHS[pool]
    if not chosen_path.exists():
        raise FileNotFoundError(
            f"qrels file not found: {chosen_path} "
            f"(pool={pool!r}; pass --qrels-path or run the producing CLI)"
        )
    qrels = load_qrels(chosen_path)

    # Bucket per-cell PRFs by (qa_id, setting, class).
    buckets: dict[tuple[str, str, str], list[PRF]] = {}
    for qa_id, sid, cls, predicted in _iter_submission_cells(submission):
        for setting in _SETTINGS:
            positives = qrels.positives(qa_id, sid, cls, setting=setting, source=source)
            res = _prf(predicted, positives, unjudged_as_zero=unjudged_as_zero)
            if res is not None:
                buckets.setdefault((qa_id, setting, cls), []).append(res)

    # Reshape into the nested dict the dataclass exposes.
    by_topic: dict[str, dict[str, dict[str, TopicMetrics]]] = {}
    for (qa_id, setting, cls), cells in buckets.items():
        bucket_p = _mean([c.P for c in cells])
        bucket_r = _mean([c.R for c in cells])
        bucket_f = _mean([c.F1 for c in cells])
        by_topic.setdefault(qa_id, {}).setdefault(setting, {})[cls] = TopicMetrics(
            P=bucket_p, R=bucket_r, F1=bucket_f, n_cells=len(cells),
        )

    # Ensure every topic has every (setting, class) slot populated so
    # downstream consumers can read uniformly.
    for qa_id in list(by_topic):
        for setting in _SETTINGS:
            by_topic[qa_id].setdefault(setting, {})
            for cls in _CLASSES:
                by_topic[qa_id][setting].setdefault(
                    cls, TopicMetrics(P=0.0, R=0.0, F1=0.0, n_cells=0)
                )

    return PerTopicReport(pool=pool, qrels_path=chosen_path, topics=by_topic)


def topic_f1_delta(
    target: PerTopicReport, anchor: PerTopicReport,
    *, setting: str = "strict", cls: str = "support",
) -> dict[str, float]:
    """Return `{qa_id: target_F1 - anchor_F1}` for every qa_id present in
    *both* reports. Topics missing from either side are dropped. The
    qualitative analysis uses this to pick the largest-positive,
    nearest-zero, and largest-negative topics for the 3-topic narrative.
    """
    common = set(target.topics) & set(anchor.topics)
    out: dict[str, float] = {}
    for qa_id in common:
        try:
            t_f1 = target.topics[qa_id][setting][cls].F1
            a_f1 = anchor.topics[qa_id][setting][cls].F1
        except KeyError:
            continue
        out[qa_id] = t_f1 - a_f1
    return out


def select_three_topics(
    deltas: dict[str, float],
) -> tuple[str, str, str]:
    """Mechanical 3-topic selection: largest positive Δ, value nearest
    zero, largest negative Δ. Ties broken by ascending integer-ordered
    qa_id (so the choice is deterministic across re-runs).

    Returns ``(positive, neutral, negative)`` qa_ids.

    Raises ``ValueError`` if fewer than 3 topics have deltas.
    """
    if len(deltas) < 3:
        raise ValueError(
            f"need ≥3 topics with deltas to pick 3 representative; got {len(deltas)}"
        )

    def _qa_int(qa: str) -> int:
        try:
            return int(qa)
        except ValueError:
            return 1 << 31  # push non-numeric to the end for ties

    pos = max(deltas, key=lambda q: (deltas[q], -_qa_int(q)))
    neg = min(deltas, key=lambda q: (deltas[q], _qa_int(q)))
    neutral = min(deltas, key=lambda q: (abs(deltas[q]), _qa_int(q)))
    # Avoid the three picks colliding: if neutral == pos or neg, fall back
    # to the second-nearest zero that's not already taken.
    if neutral in {pos, neg}:
        candidates = sorted(
            (q for q in deltas if q not in {pos, neg}),
            key=lambda q: (abs(deltas[q]), _qa_int(q)),
        )
        if not candidates:
            raise ValueError("not enough distinct topics for the 3-pick rule")
        neutral = candidates[0]
    return pos, neutral, neg
