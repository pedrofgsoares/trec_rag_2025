"""Phase 2 §12.6 — pool-coverage statistical analysis.

For each variant's ``task_a_output.json`` we score against bootstrapped
sub-samples of the §2.17 expanded qrels at varying *fractions* of the
full pool (5%, 10%, 20%, 40%, 60%, 80%, 100%). Bootstrap iterations
estimate the variance under sampling noise.

The output curve (F1 vs pool fraction) lets us answer:

1. **How much of each variant's F1 is "pool-bound"?**  At fraction
   100% we score against the full §2.17 pool; at fraction 5% we score
   against a thin pool that mimics the pool-bias condition of the
   official qrels (which contains ~12% of the §2.17 positives).

2. **Does the variant rank order change with pool size?**  If a
   variant looks best at full pool but worst at thin pool, the published
   leaderboard ordering reflects pool overlap more than algorithmic
   merit.

3. **What is the *maximum achievable* F1 given the pool?**  For each
   variant, the curve plateaus near full pool — that plateau is the
   ceiling under the current expanded-pool definition.

Reports both Strict and Relaxed; per-class (support / contradict).
Output: ``reports/pool_coverage_analysis.md`` + ASCII curves.

Pure analysis: no LLM calls, no GPU, all run-artefacts already on
disk. ~5 min CPU.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import orjson

REPO = Path(__file__).resolve().parents[1]
EXPANDED_QRELS = REPO / "data/qrels/biogen2025_taskA_qrels_expanded.jsonl"
DEFAULT_VARIANTS = [
    ("phase1_baseline", "runs/20260516-134227-phase1_baseline/task_a_output.json"),
    ("allow_existing", "runs/20260519-174814-phase2_allow_existing/task_a_output.json"),
    ("no_rerank", "runs/20260519-175116-phase2_no_rerank/task_a_output.json"),
    ("bm25_rm3", "runs/20260519-200104-phase2_bm25_rm3/task_a_output.json"),
    ("starter_baseline", "runs/starter_baseline_20260514_150718/task_a_output.json"),
]
DEFAULT_FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
DEFAULT_B = 200


@dataclass(slots=True, frozen=True)
class QrelsRow:
    qa_id: str
    sentence_id: int
    pmid: str
    base_cls: str  # "support" or "contradict" (partial_* collapsed)


def load_expanded_rows(path: Path) -> list[QrelsRow]:
    rows: list[QrelsRow] = []
    partial_to_strict = {"partial_support": "support", "partial_contradict": "contradict"}
    strict_classes = {"support", "contradict"}
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = orjson.loads(line)
            cls = str(r.get("class", ""))
            base = cls if cls in strict_classes else partial_to_strict.get(cls)
            if base is None:
                continue
            if int(r.get("relevance", 1)) <= 0:
                continue
            rows.append(
                QrelsRow(
                    qa_id=str(r["qa_id"]),
                    sentence_id=int(r["sentence_id"]),
                    pmid=str(r["pmid"]),
                    base_cls=base,
                )
            )
    return rows


def load_variant_cells(
    submission_path: Path,
) -> dict[tuple[str, int, str], set[str]]:
    """Read every (qa_id, sentence_id, class) cell with the predicted PMIDs."""
    items = orjson.loads(submission_path.read_bytes())
    out: dict[tuple[str, int, str], set[str]] = {}
    for item in items:
        meta = item.get("meta_data") or item.get("metadata") or {}
        qa_id = str(meta.get("qa_id") or "")
        for sid, ans in enumerate(item.get("answer", [])):
            for key, cls in (("supported_citations", "support"),
                             ("contradicted_citations", "contradict")):
                pmids = {str(p) for p in ans.get(key) or []}
                out[(qa_id, sid, cls)] = pmids
    return out


def score_against_pool(
    variant_cells: dict[tuple[str, int, str], set[str]],
    pool_positives: dict[tuple[str, int, str], set[str]],
    *,
    unjudged_as_zero: bool = True,
) -> dict[str, float]:
    """Compute macro F1 per class against the given positives map.

    Reproduces the methodology in ``trec_biogen.eval.metrics`` but kept
    inline so we don't reload QrelsIndex per iteration.
    """
    per_class_f1: dict[str, list[float]] = {"support": [], "contradict": []}
    for (qa_id, sid, cls), predicted in variant_cells.items():
        positives = pool_positives.get((qa_id, sid, cls), set())
        if not positives:
            if unjudged_as_zero and predicted:
                per_class_f1[cls].append(0.0)
            continue
        if not predicted:
            per_class_f1[cls].append(0.0)
            continue
        tp = len(predicted & positives)
        p = tp / len(predicted)
        r = tp / len(positives)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class_f1[cls].append(f1)
    return {
        "support_f1": mean(per_class_f1["support"]) if per_class_f1["support"] else 0.0,
        "contradict_f1": mean(per_class_f1["contradict"]) if per_class_f1["contradict"] else 0.0,
    }


def positives_from_rows(rows: list[QrelsRow]) -> dict[tuple[str, int, str], set[str]]:
    out: dict[tuple[str, int, str], set[str]] = {}
    for r in rows:
        out.setdefault((r.qa_id, r.sentence_id, r.base_cls), set()).add(r.pmid)
    return out


def percentile(values: list[float], pct: float) -> float:
    """Empirical percentile (0 <= pct <= 1) of a sample."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(pct * len(s))))
    return s[idx]


def ascii_bar(value: float, width: int = 24) -> str:
    """Render a value in [0, 1] as an ASCII bar of fixed width."""
    n = max(0, min(width, int(round(value * width))))
    return "█" * n + "·" * (width - n)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pool_coverage_analysis")
    p.add_argument("--qrels", type=Path, default=EXPANDED_QRELS)
    p.add_argument("--b", type=int, default=DEFAULT_B, help="Bootstrap iterations per fraction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("reports/pool_coverage_analysis.md"))
    p.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    all_rows = load_expanded_rows(args.qrels)
    n_total = len(all_rows)
    print(f"[pool-cov] loaded {n_total} positives from {args.qrels}", flush=True)

    variants: dict[str, dict[tuple[str, int, str], set[str]]] = {}
    for label, rel_path in DEFAULT_VARIANTS:
        path = REPO / rel_path
        if not path.exists():
            print(f"[pool-cov] WARN: {label} missing at {path}", flush=True)
            continue
        variants[label] = load_variant_cells(path)
        n_predictions = sum(len(p) for p in variants[label].values())
        print(f"[pool-cov] {label}: {len(variants[label])} cells, {n_predictions} predictions", flush=True)

    if not variants:
        print("[pool-cov] no variants found — aborting", file=sys.stderr)
        return 2

    # Outer loop over fractions; inner loop bootstrap.
    # Aggregate: results[variant][fraction] = {"support_mean": ..., "support_lo": ...}
    results: dict[str, dict[float, dict[str, float]]] = {
        v: {f: {} for f in args.fractions} for v in variants
    }

    for frac in args.fractions:
        n_sample = max(1, int(round(frac * n_total)))
        per_var_sup: dict[str, list[float]] = {v: [] for v in variants}
        per_var_con: dict[str, list[float]] = {v: [] for v in variants}
        for b in range(args.b):
            sub = rng.sample(all_rows, n_sample) if n_sample < n_total else all_rows
            pool = positives_from_rows(sub)
            for v, cells in variants.items():
                m = score_against_pool(cells, pool)
                per_var_sup[v].append(m["support_f1"])
                per_var_con[v].append(m["contradict_f1"])
        for v in variants:
            sup_vals = per_var_sup[v]
            con_vals = per_var_con[v]
            results[v][frac] = {
                "support_mean": mean(sup_vals) if sup_vals else 0.0,
                "support_lo": percentile(sup_vals, 0.025),
                "support_hi": percentile(sup_vals, 0.975),
                "contradict_mean": mean(con_vals) if con_vals else 0.0,
                "contradict_lo": percentile(con_vals, 0.025),
                "contradict_hi": percentile(con_vals, 0.975),
            }
        print(
            f"[pool-cov] frac={frac:.2f} (n={n_sample}) "
            f"done [B={args.b}]", flush=True,
        )

    # Markdown rendering.
    lines: list[str] = []
    lines.append("# Pool-coverage statistical analysis — Phase 2 §12.6\n")
    lines.append(
        f"Bootstrapped recall-vs-pool-size on the §2.17 expanded qrels "
        f"({n_total} positives across "
        f"{len(positives_from_rows(all_rows))} cells). "
        f"B = {args.b} per fraction. Seed = {args.seed}."
    )
    lines.append("")
    lines.append(
        "**Reading the table**: each row is one variant at one pool fraction. "
        "*Mean ± 95% CI* are the bootstrap percentiles. As fraction → 1.0 the "
        "score approaches the variant's expanded-pool number reported in "
        "`reports/phase2_summary.md`. As fraction → small, the score approaches "
        "what we would see under a 2025-official-style thin pool that does "
        "not match the variant's pick distribution.\n"
    )
    lines.append("## Support F1 — bootstrap mean and 95% CI per pool fraction\n")
    lines.append("| Variant | " + " | ".join(f"{f:.2f}" for f in args.fractions) + " |")
    lines.append("|---|" + "|".join("---" for _ in args.fractions) + "|")
    for v in variants:
        cells = [
            f"{results[v][f]['support_mean'] * 100:5.2f} "
            f"[{results[v][f]['support_lo'] * 100:5.2f}, {results[v][f]['support_hi'] * 100:5.2f}]"
            for f in args.fractions
        ]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Contradict F1 — bootstrap mean and 95% CI per pool fraction\n")
    lines.append("| Variant | " + " | ".join(f"{f:.2f}" for f in args.fractions) + " |")
    lines.append("|---|" + "|".join("---" for _ in args.fractions) + "|")
    for v in variants:
        cells = [
            f"{results[v][f]['contradict_mean'] * 100:5.2f} "
            f"[{results[v][f]['contradict_lo'] * 100:5.2f}, {results[v][f]['contradict_hi'] * 100:5.2f}]"
            for f in args.fractions
        ]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Visual: support F1 curve (ASCII)\n")
    lines.append("Each bar is the bootstrap mean at one fraction, scaled to [0, 50 pp].")
    lines.append("")
    lines.append("```")
    for v in variants:
        lines.append(f"{v:<28}")
        for f in args.fractions:
            val = results[v][f]["support_mean"]
            bar = ascii_bar(val / 0.5)  # rescale: 50 pp = full bar
            lines.append(f"  frac {f:.2f} | {bar} | {val * 100:5.2f} pp")
        lines.append("")
    lines.append("```")

    # Pool-bias delta per variant.
    lines.append("")
    lines.append("## Pool-bias delta — support F1 from frac=0.10 to frac=1.00\n")
    lines.append(
        "If a variant's score drops sharply when we shrink the pool, the "
        "variant is 'pool-bound' (its expanded-pool F1 is partly an artefact "
        "of how its picks overlap the full pool, not pure algorithmic quality). "
        "If a variant's score is approximately flat across fractions, the "
        "variant's score is *recall-bounded* (its expanded-pool F1 is roughly "
        "what its picks would score against any plausible pool slice)."
    )
    lines.append("")
    lines.append("| Variant | F1 @ frac=0.10 | F1 @ frac=1.00 | Δ (pp) |")
    lines.append("|---|---|---|---|")
    for v in variants:
        f10 = results[v][0.10]["support_mean"] * 100
        f1 = results[v][1.00]["support_mean"] * 100
        delta = f1 - f10
        lines.append(f"| {v} | {f10:5.2f} | {f1:5.2f} | {delta:+5.2f} |")

    lines.append("")
    lines.append("## Interpretation\n")

    # Compute pool-bias delta and rank variants by it.
    sup_pool_bias = {
        v: (results[v][1.00]["support_mean"] - results[v][0.10]["support_mean"]) * 100
        for v in variants
    }
    most_pool_bound = max(sup_pool_bias, key=sup_pool_bias.get)
    least_pool_bound = min(sup_pool_bias, key=sup_pool_bias.get)
    lines.append(
        f"- Most pool-dependent variant on support: **{most_pool_bound}** "
        f"(Δ {sup_pool_bias[most_pool_bound]:+.2f} pp from 10% to 100% pool)."
    )
    lines.append(
        f"- Least pool-dependent variant on support: **{least_pool_bound}** "
        f"(Δ {sup_pool_bias[least_pool_bound]:+.2f} pp)."
    )
    lines.append("")

    # Rank order at full pool vs at thin pool.
    full_pool_ranking = sorted(
        variants.keys(),
        key=lambda v: results[v][1.00]["support_mean"],
        reverse=True,
    )
    thin_pool_ranking = sorted(
        variants.keys(),
        key=lambda v: results[v][0.10]["support_mean"],
        reverse=True,
    )
    lines.append(f"- Support F1 ranking at frac=1.00: " + ", ".join(full_pool_ranking))
    lines.append(f"- Support F1 ranking at frac=0.10: " + ", ".join(thin_pool_ranking))
    lines.append("")
    if full_pool_ranking != thin_pool_ranking:
        lines.append(
            "**Ranking instability**: the variant ordering changes between "
            "thin and full pool. This is the statistical fingerprint of "
            "pool-bias-driven F1 differences. Any leaderboard claim based "
            "on a thin pool (like the 2025 official) carries this same "
            "instability."
        )
    else:
        lines.append(
            "**Ranking stable**: same variant ordering at thin and full pool. "
            "The expanded-pool numbers are not driven by ordering noise; the "
            "variants differ by recall, not by pool-overlap luck."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[pool-cov] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
