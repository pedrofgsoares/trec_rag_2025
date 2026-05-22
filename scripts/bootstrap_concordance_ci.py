"""Phase 2 §12.1 — bootstrap 95% CI on the macro-weighted-F1 of the
588-triple concordance set.

Reads per-call records emitted by ``trec_biogen.judge.rejudge validate
--records-out``, computes the point estimate and a 95% CI via B=1000
non-parametric resampling of triples with replacement, and writes a
short markdown stub that can be folded into ``reports/llm_judge_validation.md``.

Usage::

    uv run python scripts/bootstrap_concordance_ci.py \\
        --records data/interim/validate_cot_records.jsonl \\
        --backend-label openai-gpt-4o-mini --prompt cot \\
        --b 1000 --out reports/_bootstrap_ci_cot_mini.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson

from trec_biogen.judge.validator import bootstrap_ci, score


def load_pairs(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = orjson.loads(line)
            out.append((str(r["gold"]), str(r["pred"])))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bootstrap_concordance_ci")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--backend-label", default="openai-gpt-4o-mini")
    p.add_argument("--prompt", default="cot")
    p.add_argument("--b", type=int, default=1000, help="Bootstrap iterations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    pairs = load_pairs(args.records)
    if not pairs:
        print(f"error: no records in {args.records}", file=sys.stderr)
        return 2

    print(f"[bootstrap] loaded {len(pairs)} (gold, pred) pairs", flush=True)
    result = score(pairs)
    point, lo, hi = bootstrap_ci(pairs, n_iter=args.b, seed=args.seed)

    print(f"[bootstrap] point     = {point:.4f}")
    print(f"[bootstrap] 95% CI    = [{lo:.4f}, {hi:.4f}]")
    print(f"[bootstrap] CI width  = {hi - lo:.4f}")
    print(f"[bootstrap] gate ≥ {args.threshold:.2f}: {'PASS' if point >= args.threshold else 'FAIL'}")
    print(f"[bootstrap] CI lower bound ≥ {args.threshold:.2f}: {'PASS' if lo >= args.threshold else 'FAIL'}")

    pass_point = "PASS" if point >= args.threshold else "FAIL"
    pass_ci = "PASS" if lo >= args.threshold else "FAIL"

    body = [
        f"# Bootstrap 95% CI — {args.backend_label} (--prompt {args.prompt})",
        "",
        f"- Sample: {len(pairs)} (gold, pred) pairs from the §2.15 concordance run",
        f"- Bootstrap iterations: {args.b}, seed: {args.seed}",
        f"- Gate threshold: {args.threshold:.2f}",
        "",
        "## Per-class point estimates",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for label in ("Supports", "Contradicts", "Neutral", "Not relevant"):
        m = result.per_class[label]
        body.append(
            f"| {label} | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | {m.support} |"
        )
    body += [
        "",
        "## Macro-weighted F1 with bootstrap 95% CI",
        "",
        f"- **Point estimate**: {point:.4f} ({pass_point})",
        f"- **95% CI**: [{lo:.4f}, {hi:.4f}]  (width {hi - lo:.4f})",
        f"- **Gate threshold (CI lower bound ≥ {args.threshold:.2f})**: {pass_ci}",
        "",
        "Interpretation: with 95% probability under non-parametric resampling, the "
        f"true population macro-weighted F1 lies in [{lo:.4f}, {hi:.4f}]. The "
        f"point estimate {point:.4f} {'passes' if point >= args.threshold else 'fails'} "
        f"the design-D3 gate of {args.threshold:.2f}; the CI's lower bound "
        f"{'also passes' if lo >= args.threshold else 'does not pass'} "
        "— i.e., we can claim the gate result is statistically robust to "
        "triple-level sampling noise.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(body), encoding="utf-8")
    print(f"[bootstrap] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
