"""Calibration analysis for the LLM judge (Phase 2 §12.2).

Provides:

* :func:`reliability_bins` — equal-width binning of (confidence, correctness) pairs
* :func:`expected_calibration_error` — the standard ECE metric (Guo et al., 2017)
* :func:`isotonic_regression` — pool-adjacent-violators with tie pooling
* :func:`apply_isotonic` — apply a fitted PAV mapping with linear interpolation

The PAV implementation pools ties on x before fitting — without that step,
the algorithm produces a degenerate fit when the LLM emits the same
confidence value many times (which is exactly what happens here: gpt-4o-mini
emits 0.85, 0.90 in ~90 % of cases).

The accompanying CLI lives in ``scripts/judge_calibration.py`` and only
handles I/O + markdown rendering; all the math is here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


@dataclass(slots=True)
class CalibBin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    empirical_acc: float


def reliability_bins(
    pairs: list[tuple[float, int]],
    *,
    n_bins: int = 10,
) -> list[CalibBin]:
    """Equal-width binning of ``(confidence, correct)`` pairs into ``n_bins``.

    ``confidence`` is assumed to live in ``[0, 1]``; out-of-range values are
    clipped to the closest bin. ``correct`` should be ``0`` or ``1``.
    The last bin includes the right endpoint (``1.0`` lands in the final bin).
    """
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        idx = min(n_bins - 1, max(0, int(conf * n_bins)))
        bins[idx].append((conf, correct))
    out: list[CalibBin] = []
    for i, b in enumerate(bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if not b:
            out.append(CalibBin(lo=lo, hi=hi, n=0, mean_pred=0.0, empirical_acc=0.0))
            continue
        out.append(
            CalibBin(
                lo=lo, hi=hi, n=len(b),
                mean_pred=mean(c for c, _ in b),
                empirical_acc=mean(c for _, c in b),
            )
        )
    return out


def expected_calibration_error(bins: list[CalibBin]) -> float:
    """ECE: support-weighted absolute gap between predicted and empirical accuracy.

    Reference: Guo, C. et al. (2017). *"On Calibration of Modern Neural
    Networks"*, ICML. Lower is better-calibrated; ECE ≥ 0.05 is typically
    described as substantial mis-calibration.
    """
    total = sum(b.n for b in bins)
    if total == 0:
        return 0.0
    return sum(b.n * abs(b.mean_pred - b.empirical_acc) for b in bins) / total


def isotonic_regression(
    pairs: list[tuple[float, int]],
) -> list[tuple[float, float]]:
    """Pool-adjacent-violators isotonic regression with tie pooling.

    Returns a monotone-increasing piecewise-constant mapping
    ``x → calibrated probability`` as a sorted list of ``(x_breakpoint,
    y_calibrated)``. Standard PAV algorithm (Ayer et al., 1955), with ties on
    x pooled into a single block before fitting.
    """
    if not pairs:
        return []

    # Step 1: pool ties on x.
    grouped: dict[float, list[int]] = defaultdict(list)
    for x, y in pairs:
        grouped[x].append(y)
    pooled = sorted(grouped.items())
    sums: list[float] = [float(sum(ys)) for _, ys in pooled]
    counts: list[int] = [len(ys) for _, ys in pooled]
    xs: list[float] = [x for x, _ in pooled]

    # Step 2: PAV — merge backwards while monotonicity is violated.
    i = 0
    while i < len(sums) - 1:
        if sums[i] / counts[i] > sums[i + 1] / counts[i + 1]:
            sums[i] += sums.pop(i + 1)
            counts[i] += counts.pop(i + 1)
            xs[i] = max(xs[i], xs.pop(i + 1))
            if i > 0:
                i -= 1
        else:
            i += 1
    return [(x, s / c) for x, s, c in zip(xs, sums, counts)]


def apply_isotonic(
    raw: float,
    mapping: list[tuple[float, float]],
) -> float:
    """Apply a fitted PAV mapping to a single raw confidence value.

    Uses linear interpolation between block midpoints (continuous, not step) —
    the convention used by sklearn's ``IsotonicRegression.predict`` by default.
    Out-of-range inputs are clipped to the first / last block's calibrated value.
    """
    if not mapping:
        return raw
    if raw <= mapping[0][0]:
        return mapping[0][1]
    if raw >= mapping[-1][0]:
        return mapping[-1][1]
    for i in range(len(mapping) - 1):
        x0, y0 = mapping[i]
        x1, y1 = mapping[i + 1]
        if x0 <= raw <= x1:
            if x1 == x0:
                return (y0 + y1) / 2
            t = (raw - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return mapping[-1][1]
