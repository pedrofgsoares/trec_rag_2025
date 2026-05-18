## Why

Phase 1 (archived 2026-05-18 under `openspec/changes/archive/2026-05-18-add-baseline-pipeline/`) shipped a working pipeline that reproduces the official 2025 baseline within ±0.5 F1, but our independent pipeline scored only 5.55 / 0.52 F1 — far below the published 44.34 / 4.67 (see `reports/phase1_gap_analysis.md`). The dominant Phase 1 residual error is **TREC pool bias**, not algorithmic weakness: the official qrels only contain judgements for PMIDs the baseline picked, so any pipeline that retrieves different (potentially better) PMIDs is structurally penalised. Phase 2 separates *methodological* improvements (honest measurement) from *algorithmic* improvements (better predictions). The methodological fix unlocks informative scoring for every architectural variant — without it, ablations are uninterpretable.

## What Changes

- Add an LLM-as-judge re-judgement pipeline that classifies (answer_sentence, PMID) pairs into `{Supports, Contradicts, Neutral, Not relevant}` using a configurable backend (Llama-3.1-70B-Instruct via Together.ai by default for OSS alignment; `gpt-4o-mini`/`gpt-4o` available as paid alternatives), validated against the existing 588 human-labeled triples in `data/qrels/biogen2025_taskA_qrels.jsonl` at ≥85% per-class agreement before being trusted on novel PMIDs.
- Add an **expanded qrels** artefact (`data/qrels/biogen2025_taskA_qrels_expanded.jsonl`) covering the ~1100 PMIDs our Phase 1 pipeline emitted but that are missing from the official pool, optionally extended to BM25 top-30 per (qa_id, sentence_id) for a corpus-aware pool any future variant can be scored against.
- Extend evaluation reporting to emit **dual F1**: official-pool F1 (for §6.5 reproducibility) and expanded-pool F1 (for honest cross-architecture comparison). Default remains the official pool.
- Add three cheap ablation variants of the existing pipeline as Hydra configs: `no-rerank` (skip MedCPT-CE; BM25 top-3 direct), `no-negex` (skip cue-list pre-filter; NLI over all segmented sentences), `allow-existing` (relax `existing_supported_citations` exclusion).
- Add the contradict-path specialisation variant: swap DeBERTa-MNLI for `razent/SciFive-large-Pubmed_PMC-MedNLI` (fp16, chunked). Support path stays on DeBERTa-MNLI.
- Add retrieval-side variants: `bm25-rm3` query expansion (Pyserini-native), and `hybrid-sparse-dense` via Reciprocal Rank Fusion over BM25 plus a MedCPT-Article-Encoder FAISS-CPU index on the top-5M-by-citation-frequency subset (full corpus encoding remains out of budget; 5M subset is tractable overnight on CPU).
- Add progress instrumentation: `tqdm` bars on `phases.segment_abstracts` and `nli.negation` (Phase 1's silent multi-hour phases were the worst debugging experience). Capture per-run cost / wall-clock / VRAM-peak in `metadata.yaml`.
- Add a per-variant Phase 2 summary report (`reports/phase2_summary.md`) tabulating, per variant: Phase 1 numbers, Phase 2 official-pool F1, Phase 2 expanded-pool F1, and the delta between the last two (which quantifies the pool-bias contribution per variant).
- Add a methodological note (`reports/llm_judge_validation.md`) documenting the LLM-judge agreement rate, prompt design, and any systematic biases discovered.

Non-goals for Phase 2 (deferred to Phase 3): NLI fine-tuning on SciFact / HealthVer / BioNLI; query rewriting agents (LLM in the loop for first-stage retrieval); agentic retrieval judge / verifier modules; submission to TREC BioGEN 2026.

## Capabilities

### New Capabilities
- `llm-judge`: re-judgement subsystem — prompt template, backend abstraction (Together.ai, OpenAI), concordance validation against the human pool, cost accounting, and the emission of an expanded qrels JSONL.

### Modified Capabilities
- `biogen-task-a`: introduces Hydra-selectable architectural variants (`no-rerank`, `no-negex`, `allow-existing`, `scifive-large-contradict`, `bm25-rm3`, `hybrid-sparse-dense`), each producing its own `runs/<id>/` with the standard intermediate Parquets, comparable on the dual-pool report.
- `evaluation`: adds dual-pool scoring (`--qrels-pool=official|expanded`), per-variant Phase 2 summary reporting, and capture of cost / wall-clock / VRAM-peak per run for a Pareto-frontier view.

## Impact

- New module `src/trec_biogen/judge/` (prompt builder, backend abstraction, concordance validator, expanded-qrels emitter).
- New CLI: `python -m trec_biogen.judge.rejudge` with `--backend={together,openai,openai-mini}` and `--cost-cap=$X`.
- New Hydra configs under `configs/run/phase2_*.yaml` (one per variant), composable with existing `configs/retrieval/`, `configs/rerank/`, `configs/nli/` overrides.
- Extended `configs/nli/scifive_large_contradict.yaml`; extended `configs/retrieval/bm25_rm3.yaml`, `configs/retrieval/hybrid_rrf.yaml`.
- New runtime dependencies: `together` (or `openai`) Python SDK; `faiss-cpu` already in Phase 1 `pyproject.toml`; optional `httpx` for backend abstraction.
- Estimated storage: expanded qrels ≤ 1 MB; hybrid retrieval FAISS index over 5 M docs at MedCPT-Encoder (768 dims) ≈ 15 GB on disk.
- Wall-clock budgets: ablations 1–12 h each; SciFive-large contradict ≈ 30 h per full run (proven feasible from the Phase 1 starter-kit reproduction); hybrid retrieval encoding ≈ 24 h CPU one-off + a Phase 1-shape pass thereafter.
- LLM-judge cost: ~$3–15 for the recommended 3 000-judgement scope at `gpt-4o-mini`-first-pass + `gpt-4o`-on-borderline; lower with Together.ai Llama-3.1-70B.
- Hardware budget unchanged: WSL2 with 12 GB RAM, Quadro T1000 4 GB VRAM, 37 GB BM25 index on native ext4. Sequential model loading remains the rule.
