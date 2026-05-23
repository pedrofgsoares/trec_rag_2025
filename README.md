# TREC BioGEN 2025 — Task A baseline pipeline

Local, single-laptop, OSS-only Phase-1 grounding pipeline for the TREC BioGEN
2025 Task A: per answer sentence, emit up to 3 supporting and 3 contradicting
PubMed PMIDs against the official 26.8M-doc PubMed snapshot, scored against
the 2024 + 2025 qrels under both Strict and Relaxed settings.

Design and rationale: [`openspec/changes/add-baseline-pipeline/`](openspec/changes/add-baseline-pipeline/).

## Architecture (design D2)

```
                ┌──────────────────┐  k=100   ┌────────────┐   ┌────────────┐
question +      │   Pyserini BM25  │ ───────► │ MedCPT-CE  │ ► │ DeBERTa    │ ► support pmids
sentence ───►   │  (single index)  │          │   rerank   │   │   MNLI     │
                │                  │  k=1000  └────────────┘   └────────────┘
                │                  │ ───────► segment ► NegEx ► SciFive-MedNLI ► contradict pmids
                └──────────────────┘                                                    ↓
                                                                                  selection
                                                                                  (cap 3 each,
                                                                                  dedup, contras-
                                                                                  dicts first)
                                                                                        ↓
                                                                                 submission.jsonl
```

Each phase writes a Parquet under `runs/<id>/` and can be re-run in isolation
against its upstream output (design D6). Models are loaded sequentially:
never two heavy models resident at once (4 GB VRAM ceiling).

## Setup

Operator-only prerequisites (sudo, host-Windows access) are in [`SETUP.md`](SETUP.md).

After completing `SETUP.md` §1.1–1.4:

```bash
cd /home/up746872/projects/trec_rag_2025
source .venv/bin/activate
uv pip install -e .
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

Then place the official inputs under `data/`:

```
data/topics/biogen2025_taskA_input.jsonl   # operator-supplied
data/qrels/biogen2025_taskA_qrels.jsonl    # operator-supplied
data/qrels/biogen2024_taskA_qrels.jsonl    # operator-supplied
```

## Commands

| Step | Command | Notes |
|---|---|---|
| 1. Fetch corpus | `BIOGEN_CORPUS_URL=... bash scripts/download_pubmed.sh` | ~30 GB, 30–60 min |
| 2. Build BM25 index | `bash scripts/build_indexes.sh` | overnight, ~12 h |
| 3. Vendor starter kit | `bash scripts/vendor_starter_kit.sh` | git clone |
| 4. Baseline check (gate) | `bash scripts/baseline_check.sh` | must pass; ±2 F1 of (44.34, 4.67) |
| 5. Run full pipeline | `python -m trec_biogen.pipeline.run_task_a` | 6–10 h |
| 6. Single-phase re-run | `python -m trec_biogen.pipeline.run_task_a +run.label=ablation_x` | re-uses upstream parquets when paths match |

Each run writes `runs/<id>/` with:
* `metadata.yaml` — resolved config, git SHA, hardware fingerprint
* `log.jsonl` — structured loguru log
* `retrieval_*.parquet`, `rerank_*.parquet`, `nli_*.parquet`
* `submission.jsonl`
* `metrics_2024.json`, `metrics_2025.json`
* `report.md` — leaderboard comparison

## Tests

```bash
pytest -q
```

The CI-safe suite covers the parser, topic/qrels loaders, metrics, the
NegEx cue regex, the submission writer + validator, and an end-to-end
selection+submission smoke test against the 2-topic fixture. The BM25 round-trip
test (`tests/test_bm25_roundtrip.py`) only runs when `BIOGEN_INDEX_DIR`,
`BIOGEN_SENTINEL_PMID`, and `BIOGEN_SENTINEL_TITLE` are set.

## Phase 2 — pool-aware pipeline

Phase 2 separates **methodological** from **algorithmic** improvements over
the Phase 1 baseline. The Phase 1 residual error was dominated by **TREC
pool bias** (the official qrels only contain judgements for PMIDs the
organizers' baseline picked, so any pipeline that retrieves
different-but-correct PMIDs is structurally penalised). Phase 2 adds:

* An **LLM-as-judge** rejudge pipeline (`trec_biogen.judge`) validated at
  **macro weighted F1 = 0.8944** against the 588 human-labeled triples
  with `openai-gpt-4o-mini --prompt cot` — gate threshold 0.85 per
  design D3. Strict-mode prompts capped at ~0.75; the chain-of-thought
  prompt unlocked the inferential chain. Validation + cost breakdown in
  [`reports/llm_judge_validation.md`](reports/llm_judge_validation.md).
* An **expanded qrels** artefact
  ([`data/qrels/biogen2025_taskA_qrels_expanded.jsonl`](data/qrels/biogen2025_taskA_qrels_expanded.jsonl))
  with 1297 positives (588 human + 709 LLM) across 272 cells, drop-in
  compatible with `trec_biogen.io.qrels.load_qrels`.
* **Dual-pool eval**: `--qrels-pool={official,expanded}` and
  `--source={human,llm,any}` on `eval/metrics.py`. The §6.5 reproducibility
  anchor is recovered via `--qrels-pool=expanded --source=human` (verified
  by `tests/test_metrics.py`). Headline result: the Phase 1 pipeline
  scored **5.55 / 0.52 on the official pool but 16.43 / 12.01 on the
  §2.17 expanded pool** — Δ = +10.88 / +11.49 pp, confirming pool bias
  as a structural contributor to the apparent residual error.
* **Eight ablation variants** as Hydra configs, all inheriting `phase1_baseline`
  (seven executed; `phase2_hybrid` is the outstanding one):

  | variant | what it changes | runtime |
  |---|---|---|
  | `phase2_no_rerank` | skip MedCPT-CE; BM25 top-30 → support NLI | ~1 h |
  | `phase2_no_negex` | skip NegEx; contradict NLI over ~1.9M pairs | ~10–12 h |
  | `phase2_allow_existing` | relax `existing_supported_citations` exclusion | ~1 h |
  | `phase2_scifive_large` | swap contradict NLI to SciFive-large (fp16, T5 seq2seq) | ~5 h actual |
  | `phase2_bm25_rm3` | enable Pyserini RM3 query expansion | ~2 h |
  | `phase2_bm25_rm3_llm_filtered` | LLM-filtered PRF over RM3 pseudo-relevant set | ~2 h |
  | `phase2_bm25_llm_rewrite` | first-stage LLM query rewriting (single-shot) | ~90 min |
  | `phase2_hybrid` | BM25 + Dense (MedCPT-Encoder 5M-doc FAISS), RRF fused | ~24 h encoding + ~2 h run |

  Variants compose via Hydra defaults (`phase2_scifive_large_no_negex` is
  a valid invocation pattern).

### Phase 2.5 — judge robustness extension

A second-judge rejudge of the §2.17 candidate set with
`Llama-3.3-70B-Instruct --prompt cot` (HF Inference Providers,
auto-routed to Groq) feeds a **two-judge intersection-on-contradicts pool**
at [`data/qrels/biogen2025_taskA_qrels_intersection.jsonl`](data/qrels/biogen2025_taskA_qrels_intersection.jsonl).
`eval/metrics.py` exposes it via `--qrels-pool=intersection`. The
intersection-pool column is now part of
[`reports/phase2_summary.md`](reports/phase2_summary.md); detailed
analysis is in [`reports/judge_intersection_analysis.md`](reports/judge_intersection_analysis.md)
and the per-topic deltas in [`reports/per_topic_error_analysis.md`](reports/per_topic_error_analysis.md).

### Phase 2 commands

```bash
# LLM-judge concordance gate (one-shot, ~10 min, ~$0.08 with gpt-4o-mini).
uv run python -m trec_biogen.judge.rejudge validate \
    --backend openai-mini --prompt cot \
    --qrels  data/qrels/biogen2025_taskA_qrels.jsonl \
    --topics data/topics/biogen2025_taskA_input.json \
    --index  data/indexes/pubmed_bm25

# Rejudge novel PMIDs from a submission → expanded qrels.
# Resumable: re-invoke with the same --out after a quota / cost-cap halt.
uv run python -m trec_biogen.judge.rejudge rejudge \
    --backend openai-mini --prompt cot \
    --submission runs/<phase1_or_variant_run>/task_a_output.json \
    --qrels  data/qrels/biogen2025_taskA_qrels.jsonl \
    --topics data/topics/biogen2025_taskA_input.json \
    --index  data/indexes/pubmed_bm25 \
    --out    data/qrels/biogen2025_taskA_qrels_expanded.jsonl

# Re-score every run on both pools and regenerate the summary table.
uv run python -m trec_biogen.eval.phase2_summary \
    --include phase1_baseline phase2_   # substring filter on run dir names

# Run a variant (cheap example).
uv run python -m trec_biogen.pipeline.run_task_a \
    --config-name run/phase2_no_rerank

# Resume intermediate parquets from a prior Phase 1 run when only a
# downstream phase changes (no_negex, scifive_large):
uv run python -m trec_biogen.pipeline.run_task_a \
    --config-name run/phase2_no_negex \
    +reuse_from=runs/<phase1_baseline_run_dir>
```

### Interpreting the dual-pool table

[`reports/phase2_summary.md`](reports/phase2_summary.md) carries one row per
run (scanned from `runs/*/metadata.yaml::phase2_variant`):

| variant | F1@official Sup/Con | F1@expanded Sup/Con | F1@intersection Sup/Con | Δ Sup/Con | wall-clock | VRAM | LLM-judge $ |

* **F1@official**: published §6.5 methodology. Use for direct comparison
  with the published 44.34 / 4.67 baseline and the public leaderboard.
* **F1@expanded**: same submission, scored against the single-judge
  (`gpt-4o-mini --prompt cot`) LLM-augmented pool. Use for cross-variant
  comparison free of pool bias.
* **F1@intersection**: Phase 2.5 two-judge intersection-on-contradicts
  pool (mini-cot ∩ Llama-3.3-70B-cot). Use as the conservative reading —
  Contradicts judgements that only one judge endorses do not count.
* **Δ**: positive when the expanded pool credits picks the official pool
  penalised as false positives. Large Δ ⇒ this variant's picks were
  systematically outside the official pool.

The gap analysis behind these numbers lives in
[`reports/phase1_gap_analysis.md`](reports/phase1_gap_analysis.md); the
methodology + cost breakdown for the LLM judge in
[`reports/llm_judge_validation.md`](reports/llm_judge_validation.md). The
full design rationale is in
[`openspec/changes/archive/2026-05-22-phase2-pool-aware-pipeline/`](openspec/changes/archive/2026-05-22-phase2-pool-aware-pipeline/)
and the Phase 2.5 extension in
[`openspec/changes/archive/2026-05-22-phase2-5-judge-robustness/`](openspec/changes/archive/2026-05-22-phase2-5-judge-robustness/).

### Phase 2 / 2.5 secrets

LLM-judge backends read API keys from environment variables:
`OPENAI_API_KEY` (for `--backend openai-mini` and `--backend openai`),
`TOGETHER_API_KEY` (for `--backend together`, currently mapped to
Llama-3.3-70B-Instruct-Turbo via Together's serverless tier), and
`HF_TOKEN` (for `--backend hf-llama`, the Phase 2.5 route through HF
Inference Providers auto-routed to Groq — same Llama-3.3-70B weights at a
different price/latency point). Local `.env` is git-ignored; see
[`.env.example`](.env.example).

## Known limits (Phase 1)

* **No dense retrieval** over the full corpus (4 GB VRAM, 26.8M docs). BM25
  only on the first stage, per design D1.
* **No NLI fine-tuning.** SciFive-MedNLI is loaded from a community
  checkpoint (`razent/SciFive-base-Pubmed_PMC-MedNLI`); override via
  `nli.contradict.model=<name>` if a different fine-tune is preferred.
* **No LLM-as-NLI.** Deferred to Phase 2.
* **Hard preflight floor**: pipeline fails fast if RAM < 11 GiB
  (`SETUP.md` §1.1 not done).
* **Phase-1 gate**: `Supports F1 ≥ 60` AND `Contradicts F1 ≥ 10`, strict,
  against the 2025 qrels. Misses are recorded in
  `reports/phase1_gap_analysis.md` per task 12.2.

## Layout

```
src/trec_biogen/
  ingest/parse_pubmed.py     # 3.3 — PubMed XML → JSONL
  retrieval/
    build_collection.py      # 4.1 — JSONL → Pyserini shape
    bm25.py                  # 4.3, 4.5 — search wrapper + verify_index
  rerank/cross_encoder.py    # 7.2 — MedCPT-CE
  nli/
    stance.py                # 7.3, 8.4 — DeBERTa + SciFive
    negation.py              # 8.3 — NegEx + 23 cue patterns
  pipeline/
    phases.py                # 7.1, 8.1, 8.2, 8.5
    selection.py             # 9.1, 9.2
    run_task_a.py            # Hydra orchestrator
    preflight.py             # 4.5 + RAM + CUDA check
    metadata.py              # 11.1, 11.2
    sentences.py             # scispaCy segmentation
    model_utils.py           # 7.4 — unload + empty_cache
  io/
    topics.py                # 5.1
    qrels.py                 # 5.3
    submission.py            # 9.3, 9.4
  eval/
    metrics.py               # 6.3, 10.1 — Strict/Relaxed P/R/F1
    report.py                # 10.2, 10.3 — leaderboard + phase1_pass

configs/   # Hydra configs (retrieval, rerank, nli, run)
scripts/   # operator-run shell scripts
tests/     # CI-safe pytest suite
```
