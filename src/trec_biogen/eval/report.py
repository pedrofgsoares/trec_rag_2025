"""Leaderboard-style comparison report (10.2, 10.3).

Writes a Markdown ``report.md`` next to the JSON metric output with the four
canonical rows: official baseline / CLaC / InfoLab / current run, for both
qrels years × both settings.

Also exposes ``phase1_pass`` (task 10.3): Supports F1 ≥ 60 AND Contradicts F1
≥ 10 under the strict setting against the 2025 qrels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Published numbers (Table 5 of the official 2025 overview). Source rows
# we anchor against — DO NOT silently overwrite without citing the overview.
BASELINE_NUMBERS = {
    "strict_2025": {"support_F1": 44.34, "contradict_F1": 4.67},
    # CLaC + InfoLab numbers are added here; if a number is unknown for a
    # row we leave it as None and render as "—".
}
LEADERBOARD = {
    "baseline":  {"support_F1": 44.34, "contradict_F1": 4.67},
    "CLaC":      {"support_F1": 67.74, "contradict_F1": None},
    "InfoLab":   {"support_F1": None,  "contradict_F1": 14.15},
}

PHASE1_SUPPORT_F1 = 60.0
PHASE1_CONTRADICT_F1 = 10.0


@dataclass(slots=True)
class Phase1Verdict:
    support_f1: float
    contradict_f1: float
    passed: bool


def phase1_pass(report: dict) -> Phase1Verdict:
    """Decide the Phase-1 gate from a metrics-evaluate() report (strict, 2025 qrels).

    Inputs are interpreted as fractions in [0, 1] — they are scaled to %
    before comparison against the published thresholds.
    """
    s = report["strict"]["support"]["F1"] * 100.0
    c = report["strict"]["contradict"]["F1"] * 100.0
    return Phase1Verdict(
        support_f1=s,
        contradict_f1=c,
        passed=(s >= PHASE1_SUPPORT_F1 and c >= PHASE1_CONTRADICT_F1),
    )


def render_markdown(
    *,
    report_2025: dict,
    report_2024: dict | None,
    run_label: str,
    report_expanded: dict | None = None,
) -> str:
    """Render the run leaderboard.

    ``report_expanded``: optional metrics from the same submission scored
    against the Phase 2 expanded qrels pool. When provided, an additional
    "Dual-pool comparison (Phase 2)" section pairs official vs expanded
    F1 with a delta column. Task: Phase 2 §3.5.
    """

    def fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:.2f}"

    cur_strict_25 = {
        "support_F1": report_2025["strict"]["support"]["F1"] * 100,
        "contradict_F1": report_2025["strict"]["contradict"]["F1"] * 100,
    }
    cur_relaxed_25 = {
        "support_F1": report_2025["relaxed"]["support"]["F1"] * 100,
        "contradict_F1": report_2025["relaxed"]["contradict"]["F1"] * 100,
    }

    lines: list[str] = []
    lines.append(f"# {run_label} — leaderboard comparison\n")
    lines.append("## 2025 qrels (strict)\n")
    lines.append("| System | Supports F1 | Contradicts F1 |")
    lines.append("|---|---:|---:|")
    for name, vals in LEADERBOARD.items():
        lines.append(f"| {name} | {fmt(vals['support_F1'])} | {fmt(vals['contradict_F1'])} |")
    lines.append(
        f"| **{run_label}** | **{fmt(cur_strict_25['support_F1'])}** | "
        f"**{fmt(cur_strict_25['contradict_F1'])}** |"
    )

    lines.append("\n## 2025 qrels (relaxed) — current run only\n")
    lines.append("| Supports F1 | Contradicts F1 |")
    lines.append("|---:|---:|")
    lines.append(f"| {fmt(cur_relaxed_25['support_F1'])} | {fmt(cur_relaxed_25['contradict_F1'])} |")

    if report_2024 is not None:
        lines.append("\n## 2024 qrels (current run, both settings)\n")
        lines.append("| Setting | Supports F1 | Contradicts F1 |")
        lines.append("|---|---:|---:|")
        for setting in ("strict", "relaxed"):
            lines.append(
                f"| {setting} | {fmt(report_2024[setting]['support']['F1']*100)} | "
                f"{fmt(report_2024[setting]['contradict']['F1']*100)} |"
            )

    if report_expanded is not None:
        lines.append("\n## Dual-pool comparison (Phase 2)\n")
        lines.append(
            "Official pool = published `biogen2025_taskA_qrels.jsonl` (§6.5 "
            "anchor). Expanded pool = LLM-judge augmented qrels from Phase 2 "
            "§2.16. Δ columns are positive when the expanded pool credits picks "
            "that the official pool penalised as false positives.\n"
        )
        lines.append("| Setting | Class | Official F1 | Expanded F1 | Δ |")
        lines.append("|---|---|---:|---:|---:|")
        for setting in ("strict", "relaxed"):
            for cls in ("support", "contradict"):
                off = report_2025[setting][cls]["F1"] * 100
                exp = report_expanded[setting][cls]["F1"] * 100
                lines.append(
                    f"| {setting} | {cls} | {fmt(off)} | {fmt(exp)} | {exp - off:+.2f} |"
                )

    verdict = phase1_pass(report_2025)
    flag = "PASS" if verdict.passed else "FAIL"
    lines.append("\n## Phase-1 gate\n")
    lines.append(
        f"`Supports F1 ≥ {PHASE1_SUPPORT_F1}` AND `Contradicts F1 ≥ {PHASE1_CONTRADICT_F1}` "
        f"under strict / 2025 qrels — **{flag}** "
        f"(current: {verdict.support_f1:.2f} / {verdict.contradict_f1:.2f})."
    )
    return "\n".join(lines) + "\n"


def write_report(
    *,
    report_2025_json: Path,
    report_2024_json: Path | None,
    out_md: Path,
    run_label: str,
    report_expanded_json: Path | None = None,
) -> Path:
    r25 = json.loads(Path(report_2025_json).read_text())
    r24 = json.loads(Path(report_2024_json).read_text()) if report_2024_json else None
    re = json.loads(Path(report_expanded_json).read_text()) if report_expanded_json else None
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        render_markdown(
            report_2025=r25, report_2024=r24, run_label=run_label, report_expanded=re,
        )
    )
    return out_md
