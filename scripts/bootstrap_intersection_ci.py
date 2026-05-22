"""Phase 2.5 §3.6 — bootstrap-CI on intersection-pool cell-level F1.

Re-derives every run's per-cell F1 on the intersection qrels pool,
resamples cells with replacement to bootstrap a 95% CI on the
macro-weighted F1 per (variant, class). Output goes to
``reports/judge_intersection_analysis.md`` (cell-level CIs section).

The intersection pool's Contradicts class is sparse (43 positives across
313 cells). The CIs quantify how much of the cross-variant ordering is
noise vs signal under that sparsity.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from trec_biogen.eval.metrics import (
    DEFAULT_QRELS_PATHS,
    _CLASSES,
    _SETTINGS,
    _iter_submission_cells,
    _prf,
    PRF,
)
from trec_biogen.io.qrels import load_qrels


def _per_cell_f1s(
    run_dir: Path, qrels, setting: str, cls: str, *, unjudged_as_zero: bool = True,
) -> list[float]:
    submission = run_dir / "task_a_output.json"
    fs: list[float] = []
    for qa_id, sid, c, predicted in _iter_submission_cells(submission):
        if c != cls:
            continue
        positives = qrels.positives(qa_id, sid, cls, setting=setting, source="any")
        res = _prf(predicted, positives, unjudged_as_zero=unjudged_as_zero)
        if res is not None:
            fs.append(res.F1)
    return fs


def bootstrap_mean_ci(
    xs: list[float], *, n_iter: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float, int]:
    if not xs:
        return 0.0, 0.0, 0.0, 0
    rng = random.Random(seed)
    n = len(xs)
    samples = []
    for _ in range(n_iter):
        s = sum(xs[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    point = sum(xs) / n
    lo = samples[max(0, int((alpha / 2) * n_iter))]
    hi = samples[min(n_iter - 1, int((1 - alpha / 2) * n_iter))]
    return point, lo, hi, n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
        help="Directory containing run subdirs.",
    )
    p.add_argument(
        "--pool", type=Path,
        default=DEFAULT_QRELS_PATHS["intersection"],
        help="Qrels pool to score against (default: intersection).",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("reports/judge_intersection_analysis.md"),
    )
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    qrels = load_qrels(args.pool)
    print(f"# Intersection-pool bootstrap CI ({args.pool})")
    print()

    # Discover runs by mtime then name (matches phase2_summary order).
    runs = sorted([d for d in args.runs_dir.iterdir() if d.is_dir() and (d / "task_a_output.json").exists()])

    lines: list[str] = [
        "# Phase 2.5 §3.6 — Intersection-pool bootstrap CIs",
        "",
        f"Pool: `{args.pool}` (Phase 2.5 two-judge intersection-on-contradicts; "
        f"Supports come from canonical mini-cot, Contradicts kept only when "
        f"mini-cot ∩ HF-Llama-cot agree).",
        "",
        f"Bootstrap: {args.n_iter} resamples of per-cell F1 with replacement, "
        f"seed={args.seed}, α=0.05 (95% CI).",
        "",
        "## Cell-level macro F1 with 95% bootstrap CI (strict setting)",
        "",
        "| run | class | F1 | 95% CI | n_cells |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        for cls in _CLASSES:
            xs = _per_cell_f1s(run, qrels, setting="strict", cls=cls)
            point, lo, hi, n = bootstrap_mean_ci(
                xs, n_iter=args.n_iter, seed=args.seed,
            )
            lines.append(
                f"| {run.name} | {cls} | {point*100:.2f} "
                f"| [{lo*100:.2f}, {hi*100:.2f}] | {n} |"
            )

    lines.append("")
    lines.append(
        "**Reading note:** the Contradicts class on the intersection pool has only "
        "43 positives across 313 cells (vs 363 on the union expanded pool — an 88% "
        "drop). Cross-variant differences smaller than the CI width are inside the "
        "sampling noise floor and should not be over-interpreted."
    )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
