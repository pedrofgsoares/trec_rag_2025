## Context

The TREC BioGEN 2025 Task A leaderboard, with public qrels, gives us a fixed yardstick: top Supports F1 = 67.74 (CLaC), top Contradicts F1 = 14.15 (InfoLab), official baseline = 44.34 / 4.67. Our goal is retrospective improvement against the same qrels — no live submission. The hardware is constrained: WSL2 Ubuntu 24.04 on a laptop, 7.6 GiB visible RAM (must be raised to ≈12 GB via host `.wslconfig`), Intel i7-10750H 12 threads, NVIDIA Quadro T1000 with 4 GB VRAM, 934 GB free disk. The official corpus is provided as `biogen-2025-document-collection.zip` (~26.8M PubMed docs); the starter kit indexes it via Pyserini.

Two pieces of public evidence drive the technical shape of the change: (a) InfoLab — the only team with a non-trivial Contradicts F1 — used Pyserini + SciFive (T5 trained on MedNLI); and (b) the arXiv notebook *Negation is Not Semantic* (2603.17580) shows that dense retrievers fail on negation due to "Semantic Collapse," that lexical retrieval at deep k=1000 with NegEx pre-filtering and **sentence-level** NLI yields 4.8× better contradiction detection than document-level pipelines, and that asymmetric pool depths between support and contradiction paths beat shared pipelines.

## Goals / Non-Goals

**Goals:**
- Produce a deterministic, end-to-end pipeline that, given the official input JSONL, writes a valid Task A submission JSONL in <10 hours wall-clock on the target laptop.
- Reproduce the official baseline numbers within ±2 F1 against the 2025 qrels before any optimisation, to establish a verified "0".
- Hit Phase 1 targets: Supports F1 ≥ 60 and Contradicts F1 ≥ 10 against the 2025 qrels under the strict setting.
- Keep every intermediate artifact (retrieval rows, NLI scores) on disk in a debuggable columnar format, so any module can be ablated without re-running the others.
- Stay within the 4 GB VRAM and ≈12 GB RAM budget at all times by loading models sequentially.

**Non-Goals:**
- LLM-as-NLI on the critical path. Llama 3.1 8B Q4 in CPU is technically possible but would push wall-clock past budget and is deferred to Phase 2.
- Any dense first-stage retrieval over the full PubMed corpus. Encoding 26.8M docs with this GPU is infeasible.
- NLI fine-tuning on SciFact / HealthVer / BioNLI. Deferred to Phase 3 once the deterministic pipeline is calibrated.
- Agentic layers (query rewriting, retrieval judge, verifier). Deferred to Phase 4+.
- Submission to any TREC track. This is a retrospective measurement project.
- Distributed or multi-machine execution. Single-laptop only.

## Decisions

### D1 — Lexical-only first-stage retrieval (Pyserini BM25)

Use a single Pyserini Lucene index over the official 26.8M-doc PubMed snapshot, queried with two different `k` values: `k=100` for the support path, `k=1000` for the contradiction path.

**Why:** the *Negation is Not Semantic* result and the InfoLab result both show that lexical retrieval is competitive or superior for this task. Negation cues ("no evidence of", "absence of", "did not", "no association") are surface-form features that BM25 captures and dense embeddings flatten. Building a dense PubMed index would also be infeasible on 4 GB VRAM (~weeks to encode).

**Alternatives considered:**
- *Hybrid BM25 ∪ dense (RRF)* — rejected because dense encoding of the full corpus is out of budget.
- *BM25 + RM3 expansion* — kept open for Phase 2 but excluded from baseline to keep the spine simple and debuggable.
- *bm25s (pure-Python)* — faster to set up but lacks the Lucene tooling that the starter kit assumes; rejected to stay aligned with the official baseline.

### D2 — Decoupled support and contradiction pipelines

Two parallel paths share the BM25 index but otherwise diverge: support uses k=100 → MedCPT cross-encoder rerank → SciFive/DeBERTa entailment classifier → top-3 by score; contradiction uses k=1000 → per-abstract sentence segmentation → NegEx + cue-list filter → SciFive sentence-level contradiction classifier → top-3 by BM25 rank.

**Why:** the asymmetric pool depths come from documented evidence — high precision on support requires a tight pool, high recall on contradiction requires a deep one. The arXiv paper reports that decoupling improves both classes simultaneously, while shared pipelines force a damaging trade-off. The "top-3 by BM25 rank" choice on the contradiction path is the *simplicity paradox* finding: heuristic ordering beat calibrated probability scoring.

**Alternatives considered:**
- *Shared pipeline with class head* — simpler but empirically dominated.
- *Top-3 by NLI score on contradiction path* — rejected per the simplicity paradox.

### D3 — Sentence-level NLI, not document-level

For the contradiction path, segment each candidate abstract into sentences (scispaCy `en_core_sci_sm`), run NLI on `(answer_sentence, abstract_sentence)` pairs, and aggregate to a per-abstract contradiction score (max-pool over sentences). For the support path, keep title+abstract as a single passage but truncate to 512 tokens.

**Why:** document-level NLI dilutes the negation signal across hundreds of tokens. The arXiv paper reports a 4.8× improvement in contradiction detection from this granularity shift alone — likely the single highest-leverage decision in this design.

**Alternatives considered:**
- *Sliding-window passage NLI* — heavier with marginal gain over pure sentence-level for negation.
- *Sentence-level on both paths* — kept as a Phase 2 experiment; for support, document-level entailment is already strong and faster.

### D4 — NegEx + cue-list pre-filter on the contradiction path

Before running NLI on the deep BM25 pool, filter candidate abstract sentences to those containing at least one negation cue. Cue list = `negspacy` defaults plus 23 explicit biomedical patterns: "no evidence of", "absence of", "did not", "failed to", "no association", "no significant difference", "contrary to", "in contrast to", "however", "did not find", "was not associated", "no effect", "no significant effect", "no difference", "ruled out", "refuted", "inconsistent with", "was not significant", "did not support", "no benefit", "did not improve", "did not reduce", "no correlation".

**Why:** the deep pool (k=1000) is too large to NLI exhaustively in the time budget; the filter eliminates roughly 80–90% of candidates while preserving the contradiction-relevant ones. This trades a small recall hit for an order-of-magnitude latency reduction.

**Alternatives considered:**
- *No filter, NLI everything* — out of compute budget.
- *Learned negation classifier* — overkill for Phase 1; revisit when fine-tuning starts.

### D5 — Model selection bound to 4 GB VRAM

| Role | Model | Params | VRAM (inference) |
|---|---|---|---|
| Support reranker | `ncbi/MedCPT-Cross-Encoder` | ~110M | ~700 MB, batch 8 |
| Support NLI | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | ~184M | ~700 MB, batch 16 |
| Contradiction NLI | `razent/SciFive-base-Pubmed_PMC` fine-tuned on MedNLI (or `razent/SciFive-large-...` if VRAM allows) | ~220M base / 770M large | ~1.2 GB base, ~3 GB large with batch 4 |
| Sentence segmentation | scispaCy `en_core_sci_sm` | ~50M | CPU, ~500 MB RAM |

Models are loaded sequentially per pipeline phase: never two heavy models in VRAM simultaneously. The contradiction NLI defaults to the **base** SciFive variant; the large variant is a stretch goal for Phase 2 if 4 GB proves enough for batch=2.

**Why:** these are the largest open biomedical models that fit and have direct precedent in BioGen 2025 (InfoLab used SciFive). DeBERTa-v3-base MNLI is included for support entailment because it is a stronger general NLI than SciFive on entailment-only tasks and small enough to coexist after sequential unloading.

**Alternatives considered:**
- *MedCPT bi-encoder + ANN* — rejected because it requires encoding the whole corpus.
- *MiniLM cross-encoder* — lighter but biomedical-naive; kept as a fallback if MedCPT-CE is too slow.
- *LLM-as-NLI (Llama 3.1 8B Q4)* — feasible CPU-only but would dominate wall-clock; deferred to Phase 2.

### D6 — Sequential model loading and on-disk handoff

The pipeline runs in five distinct phases, each writing its outputs to a Parquet file under `runs/<id>/`:
1. Retrieval (BM25 k=100 and k=1000) → `retrieval_support.parquet`, `retrieval_contradict.parquet`
2. Support rerank (MedCPT-CE) → `rerank_support.parquet`
3. Support NLI (DeBERTa-MNLI) → `nli_support.parquet`
4. Contradiction NLI (SciFive-MedNLI) on NegEx-filtered sentence pairs → `nli_contradict.parquet`
5. Selection + submission writer → `submission.jsonl`

Between phases, the previous model is explicitly unloaded (`del model; torch.cuda.empty_cache()`) before the next loads. Each phase can be re-run in isolation against the upstream Parquet, enabling cheap ablations.

**Why:** the 4 GB VRAM and ≈12 GB RAM ceiling forbids a single resident pipeline. On-disk handoff also satisfies the debuggability goal — every intermediate is inspectable in DuckDB or Polars.

**Alternatives considered:**
- *In-memory pipeline* — would OOM on this hardware.
- *Streaming per-sentence* — finer granularity but harder to debug; rejected for Phase 1.

### D7 — Hydra configs + run directory layout

All knobs live in `configs/*.yaml`; each pipeline invocation snapshots its resolved config and git SHA into `runs/<id>/metadata.yaml`. The run id is `YYYYMMDD-HHMMSS-<short-config-name>`.

**Why:** turns every experiment into a comparable record; required for the ablation work that follows the baseline.

### D8 — Local evaluation replicates the official metrics

The `eval` module computes per-class precision, recall, and F1 under both Strict (`Dsup` only) and Relaxed (`Dsup ∪ Dpsup`) settings against any qrels file in the official format. Output is a JSON report plus a Markdown table that mirrors the rows of Table 5 of the official overview.

**Why:** the only credible signal for "did this change help?" is the same metric used in the leaderboard. Anything else risks Goodharting on a proxy.

### D9 — Output ordering: contradictions first

In the submission writer, for each sentence, contradicting PMIDs are emitted before supporting PMIDs. Caps: ≤3 each. Global dedup across sentences within a topic — a PMID may not appear in two sentences of the same answer.

**Why:** explicit track rule. Cheap to enforce.

### D10 — Reproduce the official baseline before any improvement

The very first run after setup is the unmodified starter-kit baseline against the 2025 qrels. The change is "ready to optimise" only when the local replication is within ±2 F1 of the published baseline (44.34 / 4.67).

**Why:** without a verified "0", every later number is unfalsifiable. This is the single most important methodological gate in the change.

## Risks / Trade-offs

- **[VRAM exhaustion mid-run]** → Sequential model loading + explicit `empty_cache`; batch sizes pinned per model in config; pre-flight check that allocates and frees the largest model before phase 1.
- **[Indexing time blows past 12h]** → Build the index in background (overnight), verify integrity via doc count, and persist; never re-index unless the corpus changes.
- **[NegEx filter drops too many true contradictions]** → Track `filtered_out_count` per topic; sample 50 dropped sentences for manual review before locking the cue list.
- **[Local baseline diverges from official numbers]** → If Δ > 2 F1 either way, halt optimisation and root-cause: tokenisation, sentence segmentation, or starter-kit version drift are the usual suspects. Do not move on until reconciled.
- **[Sentence segmentation errors on biomedical abstracts]** → scispaCy is the strongest open option; track segmentation stats and spot-check on the noisiest journals.
- **[WSL2 RAM cap not raised]** → Document `.wslconfig` as a hard prerequisite in `tasks.md`; fail fast in pipeline preflight if `psutil.virtual_memory().total < 11 GiB`.
- **[Dependency drift between Pyserini, JVM, and Python]** → Pin Python 3.11, OpenJDK 21, Pyserini, and `pyserini[lightning]` extras explicitly in `pyproject.toml`; reproduce in a fresh env in CI-style smoke test.
- **[Confusing partial-support with contradiction]** → Strict eval will catch it numerically; design the threshold tuning step to optimise contradiction precision first, then support recall.
- **[BioGen 2025 qrels too small (10 topics) for stable ablation]** → Always report numbers on both 2024 and 2025 qrels; treat single-topic deltas as noise.
