# Handoff — current state (2026-05-23)

The project is at a natural stopping point. Phases 1, 2 and 2.5 are sealed and
tagged; there is no active OpenSpec change in `openspec/changes/`. This file is
the entry point for whoever picks the work up next.

## Where things stand

| Phase | Tag | Archived change | Headline |
|---|---|---|---|
| Phase 1 baseline | `phase1-baseline` (`6984e0e`) | `openspec/changes/archive/2026-05-18-add-baseline-pipeline/` | Reproduces organisers' starter within ±2 F1 on official pool |
| Phase 2 pool-aware pipeline | `phase2-baseline` (archived at `b5dc877`) | `openspec/changes/archive/2026-05-22-phase2-pool-aware-pipeline/` | LLM-judge expanded qrels (588 → 4 758 positives) at 0.8944 macro-w-F1 gate; seven of eight ablation variants executed |
| Phase 2.5 judge robustness | `phase2.5-baseline` (`7b2d50b`) | `openspec/changes/archive/2026-05-22-phase2-5-judge-robustness/` | Two-judge intersection pool (mini-cot ∩ Llama-3.3-70B-cot via HF Providers); `no_negex` survives vs Phase 1, collapses vs starter |

The full narrative lives in [`docs/phase2_report.md`](docs/phase2_report.md);
auto-regenerated summary tables in
[`reports/phase2_summary.md`](reports/phase2_summary.md) and the per-topic /
intersection analyses under [`reports/`](reports/).

## What remains (no formally-tracked work; pick up via `/opsx:propose`)

Ordered by approximate effort:

1. **k-fold cross-validated ECE for the LLM judge** — fit PAV on `qa_id`-disjoint
   folds and replace the in-sample ECE in §10.2 with a held-out estimate.
   Cheap: minutes, no API spend. Tightens the §10.2 calibration claim.
2. **Per-topic error-analysis expansion** — Phase 2.5 covered three topics
   (largest +Δ, ~zero Δ, largest −Δ vs starter). A 10-topic pass would let
   us characterise *where* the pipeline systematically wins / loses.
3. **`phase2_hybrid` ablation** — last unrun variant. BM25 + MedCPT-Article
   dense + RRF fusion via [`src/trec_biogen/retrieval/rrf.py`](src/trec_biogen/retrieval/rrf.py).
   ~24 h CPU encoding for the 5 M-doc FAISS-CPU index + ~2 h GPU pipeline run.
   Needs its own `expand-pool` pass before honest comparison (see §11 caveat
   in the report).
4. **Third-judge concordance** — extend Phase 2.5 to a third independent
   biomedical-domain backend (e.g., Mixtral-Instruct on HF Providers) over
   the same 5 398-triple candidate set. Lets us report Krippendorff α
   instead of pairwise Jaccard and derive a three-way intersection pool.
   Estimated cost: ~$2–3 at HF Providers pricing.
5. **Phase 3 — NLI fine-tuning** — QLoRA on DeBERTa-v3-base over
   SciFact + HealthVer + BioNLI (~50 k pairs, ~6 h on a free Colab GPU).
   Expected lift: 2–5 pp on Contradicts. New `openspec` change recommended.
6. **Phase 4 — Agentic retrieval** — first-stage LLM query rewriting with
   reflection / multi-turn querying. The `phase2_bm25_llm_rewrite` variant
   (§10.7) is a single-shot proxy; the agentic pattern is open. Risk: cost
   and latency.
7. **Submit to TREC BioGEN 2026** — calendar item; track call typically
   opens June. Participating is the cleanest way to escape pool bias for
   a *system's* leaderboard numbers (rather than the methodology).

## Operational defaults (carried from prior work)

- `--prompt cot` is the canonical judge prompt on every backend (passes the
  0.85 gate on `gpt-4o-mini` at 0.8982 and on `Llama-3.3-70B` at 0.9112;
  `strict` fails ~0.75 on all backends). See
  [`reports/llm_judge_validation_openai-mini_cot.md`](reports/llm_judge_validation_openai-mini_cot.md).
- Long LLM loops: `nohup setsid` for >10 min jobs, `--max-concurrent ≤ 2`
  on OpenAI tier-1, resume-mode by design (`--out` idempotent over
  `(qa_id, sentence_id, pmid)`).
- The §6.5 reproducibility anchor — `--qrels-pool=expanded --source=human`
  must still recover the published 44.34 byte-for-byte. Any change to
  `eval/metrics.py` or `io/qrels.py` should re-run
  `tests/test_metrics.py::test_source_filter_human_recovers_official_pool_numbers`.

## Setup

[`SETUP.md`](SETUP.md) covers the one-shot environment bring-up (WSL2 RAM,
JDK, pyenv 3.11, `uv venv`, scispacy wheel, PubMed download, BM25 index
build). It has not changed since Phase 1; new contributors should still
read it end-to-end before running anything.
