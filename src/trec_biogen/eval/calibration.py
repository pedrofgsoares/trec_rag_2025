"""Calibration analysis for the LLM judge (Phase 2 §12.2, Phase 2.6 §1).

Provides:

* :func:`reliability_bins` — equal-width binning of (confidence, correctness) pairs
* :func:`expected_calibration_error` — the standard ECE metric (Guo et al., 2017)
* :func:`isotonic_regression` — pool-adjacent-violators with tie pooling
* :func:`apply_isotonic` — apply a fitted PAV mapping with linear interpolation
* :func:`kfold_ece` — held-out k-fold cross-validated ECE with folds split by
  ``qa_id`` to prevent topical leakage (Phase 2.6 §1)

The PAV implementation pools ties on x before fitting — without that step,
the algorithm produces a degenerate fit when the LLM emits the same
confidence value many times (which is exactly what happens here: gpt-4o-mini
emits 0.85, 0.90 in ~90 % of cases).

The accompanying CLI lives in ``scripts/judge_calibration.py`` and only
handles I/O + markdown rendering; all the math is here.
"""

from __future__ import annotations

import hashlib
import statistics
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


def _fold_for_qa_id(qa_id: str, k: int) -> int:
    """Deterministic fold assignment by hashing ``qa_id`` into ``[0, k)``.

    SHA-1 + modulo gives a stable mapping across Python versions (unlike the
    builtin ``hash()``, whose ``str`` hashing is randomised per-process).
    """
    digest = hashlib.sha1(qa_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % k


def kfold_ece(
    records: list[dict],
    *,
    k: int = 5,
    n_bins: int = 10,
    seed: int = 0,  # noqa: ARG001 — reserved for future stochastic variants
) -> dict[str, float | list[int]]:
    """Held-out k-fold cross-validated Expected Calibration Error.

    Splits ``records`` into ``k`` folds by ``qa_id`` (every triple from one
    topic lands in the same fold), fits the isotonic-PAV mapping on the
    ``k-1`` training folds, predicts on the held-out fold, and aggregates the
    per-fold ECE values.

    Each record SHALL contain ``qa_id`` (str-coercible), ``confidence`` (float
    in ``[0, 1]``), ``gold`` and ``pred`` (label strings). Records with a
    confidence that is not finite are skipped.

    Returns a dict with:

    * ``ece_raw_mean``, ``ece_raw_std`` — per-fold raw (uncalibrated) ECE
      mean and population standard deviation. Raw ECE is fit-free, so the
      held-out mean equals the in-sample value to within rounding noise (we
      report it for completeness and as a regression check).
    * ``ece_calibrated_mean``, ``ece_calibrated_std`` — the same for ECE
      after applying the PAV mapping fitted on the training folds.
    * ``n_per_fold`` — list of held-out triple counts per fold.
    * ``k`` — number of folds actually used (may differ from the requested
      ``k`` if fewer distinct ``qa_id`` values are present).

    Folds are assigned deterministically by ``sha1(qa_id) mod k``; the
    ``seed`` parameter is currently unused but reserved for future variants
    that shuffle topics across folds.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if not records:
        return {
            "ece_raw_mean": 0.0, "ece_raw_std": 0.0,
            "ece_calibrated_mean": 0.0, "ece_calibrated_std": 0.0,
            "n_per_fold": [], "k": 0,
        }

    # Bucket records by fold (assigned via the qa_id hash).
    folds: list[list[tuple[float, int]]] = [[] for _ in range(k)]
    for r in records:
        conf = float(r.get("confidence", 0.0))
        if not (0.0 <= conf <= 1.0):
            continue
        correct = 1 if r.get("gold") == r.get("pred") else 0
        fold = _fold_for_qa_id(str(r["qa_id"]), k)
        folds[fold].append((conf, correct))

    # Drop empty folds (happens when k > distinct qa_id count).
    populated = [(i, f) for i, f in enumerate(folds) if f]
    effective_k = len(populated)
    if effective_k < 2:
        # Cannot do CV with <2 populated folds; fall back to single-pass.
        all_pairs = [p for _, f in populated for p in f]
        bins = reliability_bins(all_pairs, n_bins=n_bins)
        raw = expected_calibration_error(bins)
        mapping = isotonic_regression(all_pairs)
        cal_pairs = [(apply_isotonic(c, mapping), y) for c, y in all_pairs]
        cal = expected_calibration_error(reliability_bins(cal_pairs, n_bins=n_bins))
        return {
            "ece_raw_mean": raw, "ece_raw_std": 0.0,
            "ece_calibrated_mean": cal, "ece_calibrated_std": 0.0,
            "n_per_fold": [len(all_pairs)], "k": 1,
        }

    raw_eces: list[float] = []
    cal_eces: list[float] = []
    n_per_fold: list[int] = []
    for held_out_idx, held_out in populated:
        train = [p for i, f in populated for p in f if i != held_out_idx]
        # Raw ECE on held-out fold.
        raw_eces.append(
            expected_calibration_error(reliability_bins(held_out, n_bins=n_bins))
        )
        # Fit on train, predict on held-out.
        mapping = isotonic_regression(train)
        cal_held = [(apply_isotonic(c, mapping), y) for c, y in held_out]
        cal_eces.append(
            expected_calibration_error(reliability_bins(cal_held, n_bins=n_bins))
        )
        n_per_fold.append(len(held_out))

    return {
        "ece_raw_mean": mean(raw_eces),
        "ece_raw_std": statistics.pstdev(raw_eces) if len(raw_eces) > 1 else 0.0,
        "ece_calibrated_mean": mean(cal_eces),
        "ece_calibrated_std": statistics.pstdev(cal_eces) if len(cal_eces) > 1 else 0.0,
        "n_per_fold": n_per_fold,
        "k": effective_k,
    }
