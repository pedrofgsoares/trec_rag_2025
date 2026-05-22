"""Phase 2 §12.2 — LLM-judge confidence calibration.

Reads the per-call records dumped by ``validate --records-out`` for one
or more backends, computes the reliability diagram (predicted-prob vs
empirical accuracy), Expected Calibration Error (ECE), and fits an
isotonic regression that maps raw model confidence to calibrated
probabilities. Outputs a markdown report at
``reports/llm_judge_calibration.md`` with per-backend reliability
tables.

A well-calibrated judge has *predicted probability ≈ empirical accuracy*
in every confidence bin. Mis-calibration usually shows up in two forms:

* **Over-confident**: emits high confidence (0.9+) on triples where it
  is wrong → reliability curve below the diagonal in the high-confidence
  region.
* **Under-confident**: emits low confidence (~0.5-0.7) on triples where
  it is right → reliability curve above the diagonal in the
  low-confidence region.

Pure analysis — runs on the existing JSONL records, no LLM calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

import orjson

from trec_biogen.eval.calibration import (
    CalibBin,
    apply_isotonic,
    expected_calibration_error,
    isotonic_regression,
    reliability_bins,
)


REPO = Path(__file__).resolve().parents[1]


def load_pairs(path: Path) -> list[tuple[float, int]]:
    """Return (confidence, correct ∈ {0, 1}) per triple."""
    out: list[tuple[float, int]] = []
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = orjson.loads(line)
            conf = float(r.get("confidence", 0.0))
            correct = 1 if r.get("gold") == r.get("pred") else 0
            out.append((conf, correct))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="judge_calibration")
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--out", type=Path, default=REPO / "reports/llm_judge_calibration.md")
    args = p.parse_args(argv)

    if len(args.records) != len(args.labels):
        print("error: --records and --labels must have same count", file=sys.stderr)
        return 2

    body: list[str] = []
    body.append("# LLM-Judge Calibration — Phase 2 §12.2\n")
    body.append(
        "Reliability diagrams + isotonic-calibration fit for the two CoT "
        "backends on the 588-triple human concordance set. Reads the per-call "
        "records dumped by `validate --records-out`. ECE is the standard "
        "expected-calibration-error metric: lower is better-calibrated.\n"
    )

    for rec_path, label in zip(args.records, args.labels):
        path = Path(rec_path)
        pairs = load_pairs(path)
        if not pairs:
            print(f"[calib] no records in {path}", file=sys.stderr)
            continue
        n = len(pairs)
        bins = reliability_bins(pairs, n_bins=args.bins)
        ece = expected_calibration_error(bins)
        # Build the calibration mapping.
        mapping = isotonic_regression(pairs)
        # Compute ECE *after* applying the mapping for a sanity check.
        calibrated_pairs = [(apply_isotonic(c, mapping), y) for c, y in pairs]
        calib_bins = reliability_bins(calibrated_pairs, n_bins=args.bins)
        ece_after = expected_calibration_error(calib_bins)

        body.append(f"\n## {label}\n")
        body.append(
            f"- Triples: {n}\n"
            f"- Overall accuracy: {mean(c for _, c in pairs):.4f}\n"
            f"- Mean raw confidence: {mean(c for c, _ in pairs):.4f}\n"
            f"- **ECE (raw)**: {ece:.4f}\n"
            f"- **ECE (after isotonic fit)**: {ece_after:.4f}\n"
        )
        body.append("### Reliability diagram (raw confidences)\n")
        body.append("| Bin (raw conf range) | n | mean pred | empirical acc | gap |")
        body.append("|---|---:|---:|---:|---:|")
        for b in bins:
            if b.n == 0:
                body.append(f"| [{b.lo:.2f}, {b.hi:.2f}) | 0 | — | — | — |")
                continue
            gap = b.mean_pred - b.empirical_acc
            body.append(
                f"| [{b.lo:.2f}, {b.hi:.2f}) | {b.n} "
                f"| {b.mean_pred:.4f} | {b.empirical_acc:.4f} | {gap:+.4f} |"
            )

        body.append("\n### ASCII reliability curve (raw vs empirical)\n")
        body.append("```")
        body.append(f"{'bin':<14} {'n':>4} {'pred':>6} {'acc':>6}  curve")
        for b in bins:
            if b.n == 0:
                continue
            width = 30
            pred_x = int(round(b.mean_pred * width))
            acc_x = int(round(b.empirical_acc * width))
            line = list("·" * (width + 1))
            line[pred_x] = "P"
            if acc_x != pred_x:
                line[acc_x] = "A"
            else:
                line[acc_x] = "X"  # both
            body.append(
                f"[{b.lo:.2f},{b.hi:.2f}) {b.n:>4} {b.mean_pred:>6.3f} "
                f"{b.empirical_acc:>6.3f}  {''.join(line)}"
            )
        body.append("                                          P = predicted prob (model)")
        body.append("                                          A = empirical accuracy")
        body.append("                                          X = aligned (P == A)")
        body.append("```")

        body.append("\n### Isotonic calibration fit\n")
        body.append(
            f"Pool-adjacent-violators (PAV) fit. {len(mapping)} monotone "
            "blocks. Apply with `apply_isotonic(raw_confidence, mapping)`.\n"
        )
        # Compress mapping to ~10 representative breakpoints for the table.
        if len(mapping) <= 10:
            picks = mapping
        else:
            step = max(1, len(mapping) // 10)
            picks = mapping[::step][:10]
            if mapping[-1] not in picks:
                picks.append(mapping[-1])
        body.append("| raw conf ≤ | calibrated prob |")
        body.append("|---|---|")
        for x, y in picks:
            body.append(f"| {x:.4f} | {y:.4f} |")

    body.append("\n## Interpretation\n")
    body.append(
        "If ECE (raw) is *substantial* (≥ 0.05), the model's emitted "
        "confidences are not interchangeable with true probabilities; "
        "use the isotonic-calibrated values when applying a confidence "
        "threshold downstream (e.g., for two-backend agreement floors "
        "or selective rejudgment).\n\n"
        "If ECE (after isotonic) is ≪ ECE (raw), the fit recovered "
        "meaningful calibration structure. If they are similar, the raw "
        "confidences are already approximately well-calibrated — common "
        "for modern instruction-tuned LLMs on simple binary-ish tasks."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(body), encoding="utf-8")
    print(f"[calib] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
