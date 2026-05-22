"""Unit tests for eval.calibration (Phase 2 §12.2)."""

from __future__ import annotations

import pytest

from trec_biogen.eval.calibration import (
    apply_isotonic,
    expected_calibration_error,
    isotonic_regression,
    reliability_bins,
)


# ----- reliability_bins ---------------------------------------------------


def test_reliability_bins_partitions_correctly() -> None:
    pairs = [(0.05, 1), (0.15, 0), (0.25, 1), (0.95, 1)]
    bins = reliability_bins(pairs, n_bins=10)
    assert len(bins) == 10
    assert bins[0].n == 1 and bins[0].mean_pred == pytest.approx(0.05)
    assert bins[1].n == 1 and bins[1].empirical_acc == 0.0
    assert bins[2].n == 1 and bins[2].empirical_acc == 1.0
    assert bins[9].n == 1
    # Empty bins.
    assert bins[3].n == 0 and bins[3].mean_pred == 0.0


def test_reliability_bins_handles_confidence_one() -> None:
    """conf == 1.0 lands in the last bin, not out-of-range."""
    bins = reliability_bins([(1.0, 1)], n_bins=5)
    assert bins[-1].n == 1
    assert bins[-1].empirical_acc == 1.0


def test_reliability_bins_empty_input() -> None:
    bins = reliability_bins([], n_bins=5)
    assert all(b.n == 0 for b in bins)


# ----- expected_calibration_error -----------------------------------------


def test_ece_perfect_calibration() -> None:
    """Pred prob == empirical acc in every bin → ECE = 0."""
    pairs = [(0.9, 1)] * 9 + [(0.9, 0)]  # 10 items, 90% accuracy at 90% pred
    ece = expected_calibration_error(reliability_bins(pairs, n_bins=10))
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_constant_offset() -> None:
    """All predictions at 0.5 but 100% correct → ECE = 0.5."""
    pairs = [(0.5, 1) for _ in range(20)]
    ece = expected_calibration_error(reliability_bins(pairs, n_bins=10))
    assert ece == pytest.approx(0.5)


def test_ece_known_pattern() -> None:
    """Asymmetric: half at conf 0.9 with acc 0.5, half at conf 0.3 with acc 0.5.
    Gap is 0.4 in one bin and 0.2 in the other → weighted mean 0.3 (equal n)."""
    pairs = [(0.9, 1), (0.9, 0)] * 5 + [(0.3, 1), (0.3, 0)] * 5
    ece = expected_calibration_error(reliability_bins(pairs, n_bins=10))
    assert ece == pytest.approx(0.3)


def test_ece_empty() -> None:
    assert expected_calibration_error([]) == 0.0


# ----- isotonic_regression (PAV with tie pooling) -------------------------


def test_pav_identity_monotone() -> None:
    """Input already monotone with unique x: output preserves order, ties pooled."""
    pairs = [(0.1, 0), (0.3, 0), (0.5, 1), (0.7, 1), (0.9, 1)]
    mapping = isotonic_regression(pairs)
    xs = [x for x, _ in mapping]
    ys = [y for _, y in mapping]
    assert xs == sorted(xs)
    # Monotone non-decreasing.
    for a, b in zip(ys, ys[1:]):
        assert a <= b


def test_pav_pools_ties_on_x() -> None:
    """Many items at the same x → one block; output prob = mean(y) at that x."""
    pairs = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
    mapping = isotonic_regression(pairs)
    assert len(mapping) == 1
    assert mapping[0] == (0.5, pytest.approx(0.5))


def test_pav_merges_violators_after_tie_pool() -> None:
    """After tie pooling, adjacent blocks violating monotonicity get merged."""
    # x=0.3 has avg y = 1.0; x=0.7 has avg y = 0.0 (anti-monotone).
    # PAV should merge them into one block with avg y = 0.5.
    pairs = [(0.3, 1), (0.3, 1), (0.7, 0), (0.7, 0)]
    mapping = isotonic_regression(pairs)
    assert len(mapping) == 1
    # Block x is the larger of the merged (0.7 > 0.3).
    assert mapping[0][0] == 0.7
    assert mapping[0][1] == pytest.approx(0.5)


def test_pav_recovers_calibration_pattern() -> None:
    """The pattern we saw in production: emit conf 0.6 → always wrong;
    conf 0.7 → 25% correct; conf 0.9 → 95% correct. After PAV the output
    is monotone with reasonable midpoints."""
    pairs = (
        [(0.6, 0)] * 10
        + [(0.7, 1)] * 5 + [(0.7, 0)] * 15
        + [(0.9, 1)] * 95 + [(0.9, 0)] * 5
    )
    mapping = isotonic_regression(pairs)
    assert len(mapping) == 3
    by_x = dict(mapping)
    assert by_x[0.6] == pytest.approx(0.0)
    assert by_x[0.7] == pytest.approx(0.25, abs=0.01)
    assert by_x[0.9] == pytest.approx(0.95, abs=0.01)


def test_pav_empty_input() -> None:
    assert isotonic_regression([]) == []


def test_pav_recovers_ece_to_near_zero_on_artificial_data() -> None:
    """End-to-end sanity: ECE before PAV >> ECE after applying PAV mapping."""
    pairs = (
        [(0.6, 0)] * 10
        + [(0.8, 1)] * 80 + [(0.8, 0)] * 20
        + [(0.95, 1)] * 95 + [(0.95, 0)] * 5
    )
    ece_raw = expected_calibration_error(reliability_bins(pairs, n_bins=10))
    mapping = isotonic_regression(pairs)
    calibrated = [(apply_isotonic(c, mapping), y) for c, y in pairs]
    ece_after = expected_calibration_error(reliability_bins(calibrated, n_bins=10))
    assert ece_after < ece_raw
    # With clean tie-structure the post-PAV ECE should be very small.
    assert ece_after < 0.01


# ----- apply_isotonic -----------------------------------------------------


def test_apply_isotonic_left_clip() -> None:
    mapping = [(0.5, 0.2), (0.8, 0.9)]
    assert apply_isotonic(0.0, mapping) == 0.2
    assert apply_isotonic(0.5, mapping) == 0.2


def test_apply_isotonic_right_clip() -> None:
    mapping = [(0.5, 0.2), (0.8, 0.9)]
    assert apply_isotonic(0.95, mapping) == 0.9
    assert apply_isotonic(0.8, mapping) == 0.9


def test_apply_isotonic_linear_interpolation_midpoint() -> None:
    mapping = [(0.5, 0.2), (0.8, 0.9)]
    # x = 0.65 is the midpoint between 0.5 and 0.8 → y is the midpoint of 0.2 and 0.9.
    assert apply_isotonic(0.65, mapping) == pytest.approx(0.55)


def test_apply_isotonic_empty_mapping_returns_raw() -> None:
    assert apply_isotonic(0.5, []) == 0.5
