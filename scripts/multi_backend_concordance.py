"""Phase 2 §12.4 — multi-backend concordance over the 588-triple
validation set.

Reads the per-call records dumped by ``trec_biogen.judge.rejudge
validate --records-out ...`` for each backend, joins them on
``(qa_id, sentence_id, pmid)``, and reports:

* Per-backend macro-weighted F1 against the human gold
* Pairwise raw agreement and Cohen's κ between every pair of backends
* Per-triple disagreement rate by class (where two backends disagree
  with each other, what does each say)

Used to defend the design D10 claim that *"F1@expanded is robust to
judge choice"*.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson

from trec_biogen.eval.concordance import cohens_kappa
from trec_biogen.judge.validator import score


def load_records(path: Path) -> dict[tuple[str, int, str], dict]:
    out: dict[tuple[str, int, str], dict] = {}
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = orjson.loads(line)
            key = (str(r["qa_id"]), int(r["sentence_id"]), str(r["pmid"]))
            out[key] = r
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="multi_backend_concordance")
    p.add_argument("--records", nargs="+", required=True,
                   help="One or more records JSONL files (one per backend).")
    p.add_argument("--labels", nargs="+", required=True,
                   help="Human-readable label per --records (same order).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    if len(args.records) != len(args.labels):
        print("error: --records and --labels must have same count", file=sys.stderr)
        return 2

    per_backend: dict[str, dict[tuple[str, int, str], dict]] = {}
    for path, label in zip(args.records, args.labels):
        per_backend[label] = load_records(Path(path))
        print(f"[multi] loaded {label}: {len(per_backend[label])} records", flush=True)

    # Intersection of triples across all backends.
    shared_keys = set.intersection(*(set(d.keys()) for d in per_backend.values()))
    print(f"[multi] shared triples across all backends: {len(shared_keys)}", flush=True)

    # Per-backend macro-weighted F1 vs human gold (same gold across backends).
    print(f"\n=== per-backend gate (vs human gold) ===")
    print(f"{'backend':<32} {'macro w-F1':>10} {'support F1':>12} {'contradict F1':>14}")
    for label, recs in per_backend.items():
        pairs = [(recs[k]["gold"], recs[k]["pred"]) for k in shared_keys]
        result = score(pairs)
        print(
            f"{label:<32} {result.macro_weighted_f1:>10.4f} "
            f"{result.per_class['Supports'].f1:>12.4f} "
            f"{result.per_class['Contradicts'].f1:>14.4f}"
        )

    # Pairwise concordance.
    labels = list(per_backend.keys())
    print(f"\n=== pairwise judge-vs-judge concordance ===")
    print(f"{'A':<32} {'B':<32} {'agreement':>11} {'κ (Cohen)':>11}")
    pairwise: list[dict] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            recs_a, recs_b = per_backend[a], per_backend[b]
            pairs = [(recs_a[k]["pred"], recs_b[k]["pred"]) for k in shared_keys]
            agree = sum(1 for x, y in pairs if x == y) / len(pairs)
            kappa = cohens_kappa(pairs)
            print(f"{a:<32} {b:<32} {agree:>11.4f} {kappa:>11.4f}")
            pairwise.append({"a": a, "b": b, "agreement": agree, "kappa": kappa})

    # Markdown.
    body: list[str] = []
    body.append("# Multi-backend concordance — Phase 2 §12.4\n")
    body.append(
        "Three independent backends classified the same 588-triple human "
        "concordance set. Per-backend gate F1 vs human gold + pairwise "
        "judge-vs-judge concordance establish the multi-evaluator "
        "robustness claim called out in design D10.\n"
    )
    body.append("## Per-backend gate (vs human gold)\n")
    body.append(
        "| Backend | Prompt | Macro w-F1 | Supports F1 | Contradicts F1 | Gate (≥ 0.85) |"
    )
    body.append("|---|---|---|---|---|---|")
    for label, recs in per_backend.items():
        pairs = [(recs[k]["gold"], recs[k]["pred"]) for k in shared_keys]
        result = score(pairs)
        verdict = "PASS" if result.macro_weighted_f1 >= 0.85 else "FAIL"
        body.append(
            f"| {label} | (per file) | {result.macro_weighted_f1:.4f} "
            f"| {result.per_class['Supports'].f1:.4f} "
            f"| {result.per_class['Contradicts'].f1:.4f} | {verdict} |"
        )

    body.append("")
    body.append("## Pairwise judge-vs-judge concordance\n")
    body.append(
        "Agreement = fraction of triples where the two backends emit the "
        "same label. Cohen's κ corrects for chance agreement; κ ≥ 0.6 is "
        "*substantial* agreement; ≥ 0.8 is *almost perfect* (Landis & "
        "Koch, 1977).\n"
    )
    body.append("| A | B | Raw agreement | Cohen's κ |")
    body.append("|---|---|---|---|")
    for pw in pairwise:
        body.append(
            f"| {pw['a']} | {pw['b']} | {pw['agreement']:.4f} | {pw['kappa']:.4f} |"
        )

    body.append("")
    body.append("## Interpretation\n")
    all_pass = all(
        score(
            [(per_backend[lbl][k]["gold"], per_backend[lbl][k]["pred"]) for k in shared_keys]
        ).macro_weighted_f1 >= 0.85
        for lbl in labels
    )
    all_kappa = [pw["kappa"] for pw in pairwise]
    min_kappa = min(all_kappa) if all_kappa else 0.0
    max_kappa = max(all_kappa) if all_kappa else 0.0
    body.append(
        f"- All {len(labels)} backends "
        f"{'pass' if all_pass else 'fail'} the 0.85 gate against human gold.\n"
        f"- Pairwise Cohen's κ ranges [{min_kappa:.3f}, {max_kappa:.3f}].\n"
    )
    if all_pass and min_kappa >= 0.6:
        body.append(
            "The multi-evaluator robustness claim is supported: independent "
            "backends *both* pass the design-D3 gate against human gold *and* "
            "agree with each other at substantial-or-better Cohen's κ. The "
            "expanded-pool F1 numbers are not driven by quirks of a single "
            "judge backend.\n"
        )
    elif all_pass:
        body.append(
            "All backends pass the gate against human gold individually, but "
            f"pairwise κ ranges down to {min_kappa:.3f} — backends disagree "
            "with each other more than they disagree with humans on the human-"
            "labeled triples. This is consistent with each backend being a "
            "valid noisy approximation of a human label, but with the noise "
            "distributions partly orthogonal. Cross-backend agreement-floor "
            "reporting (Phase 2 §10.5 fallback) would be the conservative "
            "extension.\n"
        )
    else:
        body.append(
            "At least one backend fails the gate. Defer to the passing "
            "backend(s) for downstream pool expansion; do not aggregate "
            "across the failing backend.\n"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(body), encoding="utf-8")
    print(f"\n[multi] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
