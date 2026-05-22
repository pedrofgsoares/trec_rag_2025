"""Phase 2 dual-pool summary CLI.

Scans ``runs/`` for completed runs (``task_a_output.json`` present),
re-scores each on the official and the expanded pool, joins the cost /
wall-clock / VRAM metadata captured by Phase 2 §1, and writes
``reports/phase2_summary.md`` with one row per variant per design D9::

    | variant | F1@official (Sup/Con) | F1@expanded (Sup/Con) | Δ official→expanded | wall-clock | VRAM | LLM-judge $ |

Conventions:

* The variant key is ``metadata.yaml::phase2_variant`` when present,
  else ``metadata.yaml::config.run.label``, else the dir name. This
  means the Phase 1 baseline run (no ``phase2_variant`` field) shows
  up as the "Phase 2 starting line" automatically.
* Runs without ``metadata.yaml`` (e.g. the starter-kit one-shot dump)
  are still scored — they appear as ``unknown / <dirname>``.
* The expanded-pool column is left blank if the expanded qrels file
  is missing (i.e. Phase 2 §2.16 hasn't been run yet).

Task: Phase 2 §3.4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from trec_biogen.eval.metrics import DEFAULT_QRELS_PATHS, evaluate
from trec_biogen.eval.per_topic import per_topic_f1
from trec_biogen.io.qrels import load_qrels


@dataclass(slots=True)
class RunRow:
    variant: str
    run_dir: Path
    official: dict | None
    expanded: dict | None
    intersection: dict | None
    wall_clock_s: float | None
    vram_peak_gb: float | None
    judge_cost_usd: float | None


def _maybe_load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}


def _variant_label(meta: dict, run_dir: Path) -> str:
    pv = meta.get("phase2_variant")
    if pv:
        return str(pv)
    cfg = meta.get("config") or {}
    run_cfg = cfg.get("run") or {}
    label = run_cfg.get("label")
    if label:
        return f"{label} (no phase2_variant)"
    return f"{run_dir.name} (no metadata)"


def discover_runs(
    runs_dir: Path,
    *,
    include: Iterable[str] | None = None,
) -> list[Path]:
    out: list[Path] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "task_a_output.json").exists():
            continue
        if include and not any(pat in d.name for pat in include):
            continue
        out.append(d)
    return out


def score_run(
    run_dir: Path,
    *,
    official_qrels: Path,
    expanded_qrels: Path | None,
    intersection_qrels: Path | None = None,
) -> RunRow:
    submission = run_dir / "task_a_output.json"
    meta = _maybe_load_yaml(run_dir / "metadata.yaml")
    variant = _variant_label(meta, run_dir)

    off_idx = load_qrels(official_qrels)
    official = evaluate(submission, off_idx)
    expanded: dict | None = None
    if expanded_qrels is not None and expanded_qrels.exists():
        exp_idx = load_qrels(expanded_qrels)
        expanded = evaluate(submission, exp_idx)
    intersection: dict | None = None
    if intersection_qrels is not None and intersection_qrels.exists():
        int_idx = load_qrels(intersection_qrels)
        intersection = evaluate(submission, int_idx)

    return RunRow(
        variant=variant,
        run_dir=run_dir,
        official=official,
        expanded=expanded,
        intersection=intersection,
        wall_clock_s=meta.get("wall_clock_seconds_total"),
        vram_peak_gb=meta.get("vram_peak_gb_total"),
        judge_cost_usd=meta.get("judge_cost_usd"),
    )


def _f1_pair(report: dict | None, setting: str) -> tuple[str, str]:
    if report is None:
        return ("—", "—")
    s = report[setting]["support"]["F1"] * 100
    c = report[setting]["contradict"]["F1"] * 100
    return (f"{s:.2f}", f"{c:.2f}")


def _delta(a: dict | None, b: dict | None, setting: str) -> tuple[str, str]:
    if a is None or b is None:
        return ("—", "—")
    sa, ca = a[setting]["support"]["F1"] * 100, a[setting]["contradict"]["F1"] * 100
    sb, cb = b[setting]["support"]["F1"] * 100, b[setting]["contradict"]["F1"] * 100
    return (f"{sb - sa:+.2f}", f"{cb - ca:+.2f}")


def _fmt_opt_float(v: float | None, *, suffix: str = "") -> str:
    return "—" if v is None else f"{v:.2f}{suffix}"


def render_markdown(rows: list[RunRow], *, setting: str = "strict") -> str:
    # Whether to render the intersection-pool column at all.
    any_intersection = any(r.intersection is not None for r in rows)
    if any_intersection:
        header = (
            "| variant | F1@official Sup / Con | F1@expanded Sup / Con "
            "| F1@intersection Sup / Con | Δ Sup / Con (official→expanded) "
            "| wall-clock (s) | VRAM (GiB) | LLM-judge $ |"
        )
        sep = "|---|---|---|---|---|---|---|---|"
    else:
        header = (
            "| variant | F1@official Sup / Con | F1@expanded Sup / Con "
            "| Δ Sup / Con | wall-clock (s) | VRAM (GiB) | LLM-judge $ |"
        )
        sep = "|---|---|---|---|---|---|---|"

    lines = [
        f"# Phase 2 Dual-Pool Summary ({setting})",
        "",
        "All F1 numbers are sentence-level macro under the published BioGEN "
        "2025 methodology (``unjudged_as_zero=True``). Δ columns show "
        "official → expanded (positive = pool expansion lifted the F1). "
        + (
            "The `intersection` pool is the Phase 2.5 two-judge "
            "intersection-on-contradicts pool (Supports come from the "
            "canonical mini-cot; Contradicts kept only when mini-cot and "
            "Together-Llama-70B both label them positive). `n/a` = the "
            "intersection qrels file is absent."
            if any_intersection else ""
        ),
        "",
        header,
        sep,
    ]
    for r in rows:
        off_sup, off_con = _f1_pair(r.official, setting)
        exp_sup, exp_con = _f1_pair(r.expanded, setting)
        d_sup, d_con = _delta(r.official, r.expanded, setting)
        if any_intersection:
            if r.intersection is not None:
                int_sup, int_con = _f1_pair(r.intersection, setting)
            else:
                int_sup, int_con = "n/a", "n/a"
            lines.append(
                f"| {r.variant} | {off_sup} / {off_con} | {exp_sup} / {exp_con} "
                f"| {int_sup} / {int_con} | {d_sup} / {d_con} "
                f"| {_fmt_opt_float(r.wall_clock_s)} "
                f"| {_fmt_opt_float(r.vram_peak_gb)} "
                f"| {_fmt_opt_float(r.judge_cost_usd, suffix='')} |"
            )
        else:
            lines.append(
                f"| {r.variant} | {off_sup} / {off_con} | {exp_sup} / {exp_con} "
                f"| {d_sup} / {d_con} | {_fmt_opt_float(r.wall_clock_s)} "
                f"| {_fmt_opt_float(r.vram_peak_gb)} "
                f"| {_fmt_opt_float(r.judge_cost_usd, suffix='')} |"
            )
    lines.append("")
    lines.append(f"Generated from {len(rows)} run(s) under `runs/`.")
    return "\n".join(lines) + "\n"


def _pick_topic_pool(
    *, official_qrels: Path, expanded_qrels: Path, intersection_qrels: Path,
) -> tuple[str, Path]:
    """Pool selection for the by-topic table: prefer the conservative
    intersection pool, fall back to expanded, then to official."""
    if intersection_qrels.exists():
        return "intersection", intersection_qrels
    if expanded_qrels.exists():
        return "expanded", expanded_qrels
    return "official", official_qrels


def render_by_topic_table(
    rows: list[RunRow],
    *,
    setting: str,
    official_qrels: Path,
    expanded_qrels: Path,
    intersection_qrels: Path,
) -> str:
    """Render a per-topic F1 table with one row per qa_id and one column
    pair (support / contradict) per variant. Pool source = intersection
    when present, else expanded, else official.
    """
    pool, qrels_path = _pick_topic_pool(
        official_qrels=official_qrels,
        expanded_qrels=expanded_qrels,
        intersection_qrels=intersection_qrels,
    )
    # Compute per-topic reports per row.
    reports = {
        r.variant: per_topic_f1(r.run_dir, pool=pool, qrels_path=qrels_path)
        for r in rows
    }
    # Union of qa_ids across runs, sorted by integer order when possible.
    all_qa = set()
    for rep in reports.values():
        all_qa.update(rep.topics)

    def _qa_int(q: str) -> int:
        try:
            return int(q)
        except ValueError:
            return 1 << 31

    sorted_qa = sorted(all_qa, key=lambda q: (_qa_int(q), q))

    variants = [r.variant for r in rows]
    header = "| qa_id | " + " | ".join(
        f"{v} Sup / Con" for v in variants
    ) + " |"
    sep = "|---|" + "|".join(["---"] * len(variants)) + "|"
    lines = [
        f"## Per-topic F1 ({setting}, pool={pool})",
        "",
        f"Pool source: `{qrels_path}`. Each cell is the topic's mean "
        f"per-cell F1; `n` shows cells contributing. Topics absent from a "
        f"variant's submission render as `—`.",
        "",
        header, sep,
    ]
    for qa in sorted_qa:
        cells: list[str] = []
        for v in variants:
            tm = reports[v].topics.get(qa)
            if tm is None:
                cells.append("— / —")
                continue
            s_tm = tm[setting]["support"]
            c_tm = tm[setting]["contradict"]
            cells.append(
                f"{s_tm.F1*100:.2f} (n={s_tm.n_cells}) / "
                f"{c_tm.F1*100:.2f} (n={c_tm.n_cells})"
            )
        lines.append(f"| {qa} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trec_biogen.eval.phase2_summary")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument(
        "--official-qrels", type=Path, default=DEFAULT_QRELS_PATHS["official"],
    )
    p.add_argument(
        "--expanded-qrels", type=Path, default=DEFAULT_QRELS_PATHS["expanded"],
        help="Optional: omit or point at a non-existent path to skip the expanded column.",
    )
    p.add_argument(
        "--intersection-qrels", type=Path,
        default=DEFAULT_QRELS_PATHS["intersection"],
        help="Phase 2.5 two-judge intersection-on-contradicts pool. The column "
             "is hidden in the report when the file is absent — existing Phase 2 "
             "outputs are not affected.",
    )
    p.add_argument(
        "--include", nargs="*", default=None,
        help="Substring filter on run-dir names. Default: every run with task_a_output.json.",
    )
    p.add_argument(
        "--setting", choices=["strict", "relaxed"], default="strict",
    )
    p.add_argument(
        "--by-topic", action="store_true",
        help="Append a second table with one row per qa_id and columns per "
             "(variant, pool). Pool source for the topic table is the "
             "intersection qrels when available, else expanded, else official.",
    )
    p.add_argument(
        "--out", type=Path, default=Path("reports/phase2_summary.md"),
    )
    args = p.parse_args(argv)

    run_dirs = discover_runs(args.runs_dir, include=args.include)
    if not run_dirs:
        print(f"no run dirs with task_a_output.json under {args.runs_dir}", file=sys.stderr)
        return 1
    rows = [
        score_run(
            d,
            official_qrels=args.official_qrels,
            expanded_qrels=args.expanded_qrels,
            intersection_qrels=args.intersection_qrels,
        )
        for d in run_dirs
    ]
    body = render_markdown(rows, setting=args.setting)

    if args.by_topic:
        body += "\n" + render_by_topic_table(
            rows,
            setting=args.setting,
            official_qrels=args.official_qrels,
            expanded_qrels=args.expanded_qrels,
            intersection_qrels=args.intersection_qrels,
        )

    # Phase 2 §10.4: persistent commentary. If a sibling
    # `<out>_commentary.md` exists, append it verbatim under a separator
    # so regenerating the auto-table doesn't clobber the analysis prose.
    commentary_path = args.out.with_name(args.out.stem + "_commentary.md")
    if commentary_path.exists():
        commentary_text = commentary_path.read_text(encoding="utf-8")
        body = body.rstrip() + "\n\n---\n\n" + commentary_text.lstrip()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(body)
    print(f"\nwrote {args.out}", file=sys.stderr)
    if commentary_path.exists():
        print(f"(appended commentary from {commentary_path})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
