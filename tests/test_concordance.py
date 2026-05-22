"""Unit tests for eval.concordance (Phase 2 §12.4 Cohen's κ)."""

from __future__ import annotations

import pytest

from trec_biogen.eval.concordance import cohens_kappa


def test_kappa_perfect_agreement() -> None:
    pairs = [("Supports", "Supports")] * 5 + [("Contradicts", "Contradicts")] * 5
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_kappa_total_disagreement_balanced_classes() -> None:
    """Two raters always disagree on a 2-class set with equal marginals."""
    pairs = [("A", "B"), ("B", "A"), ("A", "B"), ("B", "A")]
    # p_o = 0, p_e = 0.5 (each rater outputs each label 50/50), κ = (0 - 0.5)/(1 - 0.5) = -1.0
    assert cohens_kappa(pairs) == pytest.approx(-1.0)


def test_kappa_independent_random_with_class_prior() -> None:
    """Random raters with the same class distribution → κ ≈ 0."""
    # 100 pairs where each rater independently votes Supports 80% of the time.
    # By construction p_o ≈ 0.8 * 0.8 + 0.2 * 0.2 = 0.68, p_e = same → κ ≈ 0.
    import random
    rng = random.Random(0)
    n = 2000
    pairs = [
        ("Supports" if rng.random() < 0.8 else "Contradicts",
         "Supports" if rng.random() < 0.8 else "Contradicts")
        for _ in range(n)
    ]
    k = cohens_kappa(pairs)
    assert abs(k) < 0.05   # within sampling noise of zero


def test_kappa_imbalanced_classes_falls_in_fair_range() -> None:
    """With class imbalance (549/39 Supports vs Contradicts) and high raw
    agreement, κ should land in the "fair" range (0.20-0.40 per Landis &
    Koch 1977), well below the raw agreement of ~87% because chance
    agreement is already high under the imbalance."""
    pairs: list[tuple[str, str]] = []
    pairs += [("Supports", "Supports")] * 500
    pairs += [("Supports", "Neutral")] * 49
    pairs += [("Contradicts", "Contradicts")] * 10
    pairs += [("Contradicts", "Supports")] * 25
    pairs += [("Contradicts", "Neutral")] * 4
    k = cohens_kappa(pairs)
    raw_agreement = (500 + 10) / 588   # ≈ 0.867
    # Manually computed: p_e ≈ 0.834, κ = (0.867 - 0.834) / (1 - 0.834) ≈ 0.197.
    assert k == pytest.approx(0.197, abs=0.01)
    # And κ << raw_agreement, demonstrating that class-imbalance correction matters.
    assert k < raw_agreement / 3


def test_kappa_empty_returns_zero() -> None:
    assert cohens_kappa([]) == 0.0


def test_kappa_all_one_label_both_raters() -> None:
    """Both raters always say 'Supports' → p_e = 1, return 1.0 by convention."""
    pairs = [("Supports", "Supports")] * 10
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_kappa_one_rater_constant_other_variable() -> None:
    """Rater A always Supports, rater B sometimes Contradicts → κ defined and < 1."""
    pairs = [("Supports", "Supports")] * 8 + [("Supports", "Contradicts")] * 2
    k = cohens_kappa(pairs)
    # κ here is 0 because rater A is constant (a_dist = {Supports: 1.0}),
    # so the chance-agreement on Supports equals the actual agreement.
    assert k == pytest.approx(0.0, abs=1e-9)
