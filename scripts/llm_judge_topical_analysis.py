"""Phase 2 §12.3 — per-topic / per-class distribution of LLM-emitted positives.

Reads the expanded qrels and produces:
  * per-topic counts of LLM positives (support, contradict, total)
  * comparison vs the human pool (where humans labelled, how many LLM
    positives were added on top)
  * a "support-overgeneration ratio" per topic: LLM-support /
    LLM-contradict — flags topical clusters where the judge is
    one-sided
  * topical themes for the top-K and bottom-K topics by support density,
    pulled from the BioGen 2025 topic questions
  * a markdown report at ``reports/llm_judge_topical_bias.md``

This is the diagnostic the Perplexity review recommended as task 12.3
to surface whether the judge is biased toward particular topical
clusters before relying on the expanded qrels for cross-variant
comparison.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import polars as pl

REPO = Path(__file__).resolve().parents[1]
EXPANDED = REPO / "data/qrels/biogen2025_taskA_qrels_expanded.jsonl"
TOPICS = REPO / "data/topics/biogen2025_taskA_input.json"
OUT_MD = REPO / "reports/llm_judge_topical_bias.md"


def load_expanded(path: Path) -> pl.DataFrame:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(
                {
                    "qa_id": str(r["qa_id"]),
                    "sentence_id": int(r["sentence_id"]),
                    "pmid": str(r["pmid"]),
                    "cls": str(r.get("class", "")),
                    "source": str(r.get("source", "human")),
                    "confidence": float(r.get("confidence", 1.0))
                    if r.get("confidence") is not None else 1.0,
                }
            )
    return pl.DataFrame(rows)


def topic_questions(path: Path) -> dict[str, str]:
    items = orjson.loads(path.read_bytes())
    return {
        str(it["meta_data"]["qa_id"]): it["meta_data"]["question"]
        for it in items
    }


def main() -> int:
    if not EXPANDED.exists():
        print(f"error: expanded qrels missing at {EXPANDED}", file=sys.stderr)
        return 2

    df = load_expanded(EXPANDED)
    questions = topic_questions(TOPICS)

    # Strict "human" tag includes both class=support and partial_support; collapse.
    def normalise_cls(c: str) -> str:
        if c.startswith("partial_"):
            return c[len("partial_"):]
        return c

    df = df.with_columns(pl.col("cls").map_elements(normalise_cls, return_dtype=pl.String).alias("cls"))

    n_total = df.shape[0]
    n_human = df.filter(pl.col("source") == "human").shape[0]
    n_llm = n_total - n_human
    print(
        f"[topical] total positives: {n_total}; human: {n_human}; LLM: {n_llm}",
        flush=True,
    )

    # Per-topic LLM positives by class.
    llm_only = df.filter(pl.col("source") != "human")
    by_topic = (
        llm_only.group_by(["qa_id", "cls"]).agg(pl.len().alias("n"))
        .pivot(index="qa_id", on="cls", values="n", aggregate_function="first")
        .fill_null(0)
        .with_columns(
            (pl.col("support") + pl.col("contradict")).alias("llm_total"),
        )
        .sort("llm_total", descending=True)
    )

    # Per-topic human positives — for comparison.
    human_only = df.filter(pl.col("source") == "human")
    human_by_topic = (
        human_only.group_by(["qa_id", "cls"]).agg(pl.len().alias("n_human"))
        .pivot(index="qa_id", on="cls", values="n_human", aggregate_function="first")
        .rename({"support": "human_support", "contradict": "human_contradict"})
        .fill_null(0)
    )

    # Confidence stats by topic (LLM only).
    conf_stats = (
        llm_only.group_by("qa_id").agg(
            pl.col("confidence").mean().alias("conf_mean"),
            pl.col("confidence").median().alias("conf_median"),
            pl.col("confidence").min().alias("conf_min"),
        )
    )

    joined = (
        by_topic.join(human_by_topic, on="qa_id", how="left")
        .join(conf_stats, on="qa_id", how="left")
        .with_columns(
            (pl.col("support") / (pl.col("contradict") + 1)).alias("sup_con_ratio"),
            pl.col("qa_id").map_elements(
                lambda q: questions.get(q, "?"), return_dtype=pl.String
            ).alias("question"),
        )
        .fill_null(0)
    )

    # --- markdown report ---
    sup_total = int(llm_only.filter(pl.col("cls") == "support").shape[0])
    con_total = int(llm_only.filter(pl.col("cls") == "contradict").shape[0])
    n_topics_with_llm = joined.shape[0]
    mean_sup = float(joined["support"].mean() or 0)
    median_sup = float(joined["support"].median() or 0)
    iqr_lo = float(joined["support"].quantile(0.25) or 0)
    iqr_hi = float(joined["support"].quantile(0.75) or 0)
    overall_ratio = sup_total / max(1, con_total)

    lines: list[str] = []
    lines.append("# LLM-Judge Topical Bias Analysis — Phase 2 §12.3\n")
    lines.append(
        "Per-topic distribution of the 4170 LLM-emitted positives in "
        f"`{EXPANDED.relative_to(REPO)}`. Diagnostic for whether the "
        "judge systematically overgenerates supports or concentrates "
        "positives in particular topical clusters.\n"
    )
    lines.append("## Aggregate\n")
    lines.append(f"- Topics with ≥ 1 LLM positive: **{n_topics_with_llm}**")
    lines.append(f"- LLM support positives total: **{sup_total}**")
    lines.append(f"- LLM contradict positives total: **{con_total}**")
    lines.append(f"- Global support/contradict ratio: **{overall_ratio:.2f}**")
    lines.append("")
    lines.append("Per-topic LLM-support count:")
    lines.append(f"- mean: {mean_sup:.1f}, median: {median_sup:.0f}, IQR: [{iqr_lo:.0f}, {iqr_hi:.0f}]")
    lines.append("")

    lines.append("## Top-10 topics by LLM-support density\n")
    lines.append(
        "Topics where the LLM judge accepted the most novel-support claims. "
        "High counts here may indicate either (a) a productive topical area "
        "with plentiful supporting evidence in PubMed, or (b) judge "
        "overgeneration — manual spot-check the top-3 before trusting."
    )
    lines.append("")
    lines.append("| qa_id | question (truncated) | LLM sup | LLM con | sup/con | human sup | mean conf |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in joined.head(10).iter_rows(named=True):
        q = (row["question"] or "?")[:60]
        lines.append(
            f"| {row['qa_id']} | {q} | {int(row['support'])} | {int(row['contradict'])} "
            f"| {row['sup_con_ratio']:.1f} | {int(row.get('human_support', 0))} "
            f"| {row['conf_mean']:.2f} |"
        )

    lines.append("")
    lines.append("## Bottom-10 topics by LLM-support density\n")
    lines.append(
        "Topics where the LLM judge accepted few novel supports. Either "
        "(a) genuinely narrow topical evidence in PubMed, or (b) the BM25 "
        "first-stage missed evidence for this topic and the LLM had nothing "
        "to accept."
    )
    lines.append("")
    lines.append("| qa_id | question (truncated) | LLM sup | LLM con | sup/con | human sup | mean conf |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in joined.tail(10).iter_rows(named=True):
        q = (row["question"] or "?")[:60]
        lines.append(
            f"| {row['qa_id']} | {q} | {int(row['support'])} | {int(row['contradict'])} "
            f"| {row['sup_con_ratio']:.1f} | {int(row.get('human_support', 0))} "
            f"| {row['conf_mean']:.2f} |"
        )

    lines.append("")
    lines.append("## Overgeneration check — extreme sup/con ratios\n")
    lines.append(
        "Topics where the LLM judge accepts many supports but ~zero "
        "contradicts. Cross-check whether these topics have "
        "biologically-plausible *contradiction* candidates in PubMed; "
        "if yes, the judge is biased toward `Supports` on this topical "
        "subset."
    )
    lines.append("")
    extreme = (
        joined.filter((pl.col("support") >= 30) & (pl.col("contradict") <= 1))
        .sort("support", descending=True)
    )
    if extreme.shape[0] == 0:
        lines.append("_No topics with ≥30 supports and ≤1 contradicts — judge is well-distributed._")
    else:
        lines.append("| qa_id | question | LLM sup | LLM con |")
        lines.append("|---|---|---:|---:|")
        for row in extreme.iter_rows(named=True):
            q = (row["question"] or "?")[:70]
            lines.append(
                f"| {row['qa_id']} | {q} | {int(row['support'])} | {int(row['contradict'])} |"
            )

    lines.append("")
    lines.append("## Per-class confidence distribution (LLM rows only)\n")
    sup_conf = llm_only.filter(pl.col("cls") == "support")["confidence"]
    con_conf = llm_only.filter(pl.col("cls") == "contradict")["confidence"]
    lines.append("| Class | n | mean conf | median | min | low-confidence (<0.7) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    if sup_conf.shape[0] > 0:
        n_low = int((sup_conf < 0.7).sum())
        lines.append(
            f"| Supports | {sup_conf.shape[0]} | {float(sup_conf.mean()):.3f} | "
            f"{float(sup_conf.median()):.3f} | {float(sup_conf.min()):.3f} | "
            f"{n_low} ({100 * n_low / sup_conf.shape[0]:.1f}%) |"
        )
    if con_conf.shape[0] > 0:
        n_low = int((con_conf < 0.7).sum())
        lines.append(
            f"| Contradicts | {con_conf.shape[0]} | {float(con_conf.mean()):.3f} | "
            f"{float(con_conf.median()):.3f} | {float(con_conf.min()):.3f} | "
            f"{n_low} ({100 * n_low / con_conf.shape[0]:.1f}%) |"
        )

    lines.append("")
    lines.append("## Interpretation\n")
    if overall_ratio > 15:
        lines.append(
            f"⚠️ The global LLM support/contradict ratio is **{overall_ratio:.1f}** — "
            "the judge is *strongly* skewed toward Supports. Cross-check whether "
            "the underlying BM25 top-30 candidates actually contain contradicting "
            "evidence in equal measure; if yes, the judge may be conservative on "
            "contradict labels and the expanded-pool contradict numbers are an "
            "*upper bound* on what a calibrated judge would emit."
        )
    elif overall_ratio > 8:
        lines.append(
            f"The global support/contradict ratio is **{overall_ratio:.1f}** — "
            "consistent with PubMed's known prior toward affirmative findings. "
            "Treat with mild caution but not a strong bias signal."
        )
    else:
        lines.append(
            f"The global support/contradict ratio is **{overall_ratio:.1f}** — "
            "within the expected range for biomedical evidence retrieval. "
            "No strong topical bias signal."
        )
    lines.append("")
    lines.append(
        "Data sources:\n"
        f"- expanded qrels: `{EXPANDED.relative_to(REPO)}`\n"
        f"- topic questions: `{TOPICS.relative_to(REPO)}`\n"
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[topical] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
