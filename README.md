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
