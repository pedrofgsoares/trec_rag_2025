"""Pool-coverage statistical analysis (Phase 2 §12.6).

Provides the bootstrap sub-sampling helper used by
``scripts/pool_coverage_analysis.py`` to estimate recall-vs-pool-size
curves. The function is small and pure (no I/O, no dependencies beyond
``random``) so it can be unit-tested independently of the report
rendering.

The standard usage pattern: given a list of qrels positives (one tuple
per ``(qa_id, sentence_id, class, pmid)`` row), sub-sample with the
specified fraction (without replacement) and feed the resulting subset
into ``trec_biogen.io.qrels.QrelsIndex``-style structures to rescore.

A typical bootstrap experiment runs many iterations per fraction and
reports the mean F1 plus 2.5th / 97.5th percentile CI bounds.
"""

from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def subsample(
    population: Sequence[T],
    *,
    fraction: float,
    seed: int = 0,
) -> list[T]:
    """Sample a deterministic-with-seed subset of ``population``.

    ``fraction`` ∈ ``[0, 1]``. The output size is ``round(len(population)
    * fraction)``; sampling is uniform without replacement. The same
    ``seed`` returns the same subset every call.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    n = round(len(population) * fraction)
    if n <= 0:
        return []
    if n >= len(population):
        return list(population)
    rng = random.Random(seed)
    return rng.sample(list(population), n)


def percentile_ci(
    values: list[float],
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Empirical ``1 - alpha`` confidence interval from a list of bootstrap samples.

    Returns ``(lo, hi)`` = the ``alpha/2`` and ``1 - alpha/2`` empirical
    percentiles of ``values``. ``alpha = 0.05`` gives a 95 % CI.
    """
    if not values:
        return 0.0, 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    lo_idx = max(0, int((alpha / 2) * n))
    hi_idx = min(n - 1, int((1.0 - alpha / 2) * n))
    return sorted_vals[lo_idx], sorted_vals[hi_idx]
