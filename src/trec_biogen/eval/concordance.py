"""Inter-rater concordance metrics for multi-backend LLM-judge comparisons
(Phase 2 §12.4).

Currently exposes Cohen's κ (Cohen, 1960). The function is small and
self-contained; it lives in a module rather than inline in the script so it
can be unit-tested without depending on the I/O surface of the
``multi_backend_concordance.py`` CLI.

Interpretation thresholds (Landis & Koch, 1977):

* κ < 0.00 : *poor* (worse than chance)
* 0.00 - 0.20 : *slight*
* 0.21 - 0.40 : *fair*
* 0.41 - 0.60 : *moderate*
* 0.61 - 0.80 : *substantial*
* 0.81 - 1.00 : *almost perfect*
"""

from __future__ import annotations

from collections import Counter


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's κ between two raters over the same n items.

    ``pairs`` is a list of ``(rater_a_label, rater_b_label)`` over a
    common item set. Labels can be any hashable type — typically str.
    """
    if not pairs:
        return 0.0
    n = len(pairs)
    labels = sorted({lbl for p in pairs for lbl in p})
    if not labels:
        return 0.0
    p_o = sum(1 for a, b in pairs if a == b) / n
    a_dist = Counter(a for a, _ in pairs)
    b_dist = Counter(b for _, b in pairs)
    p_e = sum((a_dist[l] / n) * (b_dist[l] / n) for l in labels)
    if p_e >= 1.0:
        # Both raters always emit the same single label — undefined; treat as
        # perfect agreement by convention (consistent with sklearn).
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)
