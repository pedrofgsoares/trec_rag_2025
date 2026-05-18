## Why

TREC BioGEN 2025 Task A asks systems to ground each sentence of a candidate biomedical answer with up to three supporting and three contradicting PubMed PMIDs. Public 2025 results show a wide gap that we can exploit: the best Supports F1 was 67.74 (CLaC) and the best Contradicts F1 was only 14.15 (InfoLab) — most teams scored zero on contradictions, and the official baseline scored 44.34 / 4.67. We have qrels and corpus in hand, so improvements are now directly measurable. We need a reproducible, locally-runnable Phase 1 pipeline that establishes a trustable baseline and unlocks targeted experimentation against the 2025 leaderboard.

## What Changes

- Add a deterministic, end-to-end pipeline that consumes the official BioGen 2025 input JSONL and produces a valid submission JSONL that scores against the 2025 qrels.
- Add ingestion + indexing of the official `biogen-2025-document-collection.zip` (≈26.8M PubMed docs) into a local Pyserini BM25 index.
- Add a **decoupled retrieval architecture**: separate support path (BM25 k=100 → MedCPT cross-encoder → SciFive/DeBERTa NLI) and contradiction path (BM25 k=1000 → sentence segmentation → NegEx pre-filter → SciFive-MedNLI sentence-level classifier).
- Add a sentence-level NLI step instead of document-level, enforced by abstract sentence segmentation before classification.
- Add a selection module that emits up to 3 supports and 3 contradictions per sentence, contradictions first per the track rule, with global PMID dedup.
- Add a local evaluation script that replicates the official Strict and Relaxed metrics (per-class P/R/F1) against BioGen 2024 and BioGen 2025 qrels.
- Add reproducibility scaffolding: Hydra configs, run directory layout (`runs/<id>/`), structured logs, and pinned dependencies.
- Pin hardware budget to the available machine (WSL2, ~12 GiB RAM after `.wslconfig` bump, 4 GB VRAM Quadro T1000, 12-thread i7-10750H, ~930 GB disk). No model loaded simultaneously may exceed VRAM; pipeline runs sequentially.

Non-goals for Phase 1: LLM-as-NLI, query rewriting agents, dense first-stage retrieval, NLI fine-tuning, ensembles. These are explicitly deferred to later phases.

## Capabilities

### New Capabilities
- `biogen-task-a`: end-to-end Task A grounding — consumes input JSONL, produces a valid submission JSONL with per-sentence support and contradiction PMIDs subject to the track's caps and ordering rule.
- `pubmed-index`: local ingestion and BM25 indexing of the official BioGen 2025 PubMed snapshot, including a build script and integrity checks.
- `evaluation`: local replication of the official Strict and Relaxed Task A metrics against published qrels, producing per-class P/R/F1 and a stable comparison report against the 2025 leaderboard rows.

### Modified Capabilities
<!-- None — this is the first change in the project. -->

## Impact

- New top-level Python package `src/trec_biogen/` with submodules `ingest`, `retrieval`, `rerank`, `nli`, `pipeline`, `io`, `eval`.
- New `configs/` tree (Hydra) and `scripts/` (download, index, run).
- New `data/` tree (gitignored): raw PubMed dump, Lucene index, parsed JSONL, qrels.
- New runtime dependencies: Python 3.11, OpenJDK 21, Pyserini, transformers, sentence-transformers, scispaCy + `en_core_sci_sm`, `negspacy`, FAISS-CPU (light use), Hydra, MLflow (local), DuckDB, polars/pandas.
- Disk footprint ≈ 100–130 GB (corpus + index + parsed + models). Fits comfortably in 934 GB free.
- Wall-clock budget for the full Phase 1 pipeline on this hardware: ≈ 12 h indexing (one-off), ≈ 6–10 h per full run over the official topic set.
- Hard prerequisite: WSL2 RAM bumped to ≥ 12 GB via host `.wslconfig`. Documented in setup.
- No external paid services. All models OSS and locally hosted.
