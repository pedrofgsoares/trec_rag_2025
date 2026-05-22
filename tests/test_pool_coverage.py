"""Unit tests for eval.pool_coverage (Phase 2 §12.6 sub-sampling helpers)."""

from __future__ import annotations

import pytest

from trec_biogen.eval.pool_coverage import percentile_ci, subsample


# ----- subsample ---------------------------------------------------------


def test_subsample_size_matches_fraction() -> None:
    pop = list(range(100))
    assert len(subsample(pop, fraction=0.1, seed=0)) == 10
    assert len(subsample(pop, fraction=0.5, seed=0)) == 50
    assert len(subsample(pop, fraction=1.0, seed=0)) == 100


def test_subsample_zero_fraction() -> None:
    assert subsample(list(range(100)), fraction=0.0, seed=0) == []


def test_subsample_deterministic_with_seed() -> None:
    pop = list(range(100))
    a = subsample(pop, fraction=0.3, seed=42)
    b = subsample(pop, fraction=0.3, seed=42)
    assert a == b


def test_subsample_different_seeds_give_different_samples() -> None:
    pop = list(range(100))
    a = subsample(pop, fraction=0.3, seed=0)
    b = subsample(pop, fraction=0.3, seed=1)
    assert a != b


def test_subsample_without_replacement() -> None:
    pop = list(range(50))
    out = subsample(pop, fraction=1.0, seed=0)
    assert sorted(out) == pop   # full population back, no duplicates


def test_subsample_rejects_out_of_range_fraction() -> None:
    with pytest.raises(ValueError):
        subsample(list(range(10)), fraction=1.5)
    with pytest.raises(ValueError):
        subsample(list(range(10)), fraction=-0.1)


def test_subsample_handles_empty_population() -> None:
    assert subsample([], fraction=0.5, seed=0) == []


# ----- percentile_ci -----------------------------------------------------


def test_percentile_ci_known_distribution() -> None:
    """For a uniform [0, 1] distribution, 95% CI from 0.025 to 0.975."""
    vals = [i / 1000 for i in range(1001)]
    lo, hi = percentile_ci(vals, alpha=0.05)
    assert lo == pytest.approx(0.025, abs=0.005)
    assert hi == pytest.approx(0.975, abs=0.005)


def test_percentile_ci_empty() -> None:
    assert percentile_ci([], alpha=0.05) == (0.0, 0.0)


def test_percentile_ci_full_alpha_returns_min_max() -> None:
    vals = [1.0, 5.0, 2.0, 9.0, 4.0]
    lo, hi = percentile_ci(vals, alpha=0.0)
    assert lo == 1.0 and hi == 9.0


def test_percentile_ci_single_value() -> None:
    lo, hi = percentile_ci([0.7], alpha=0.05)
    assert lo == 0.7 and hi == 0.7
