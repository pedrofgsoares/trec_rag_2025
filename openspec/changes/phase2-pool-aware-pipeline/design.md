## Context

Phase 1 produced a fully working, calibrated pipeline (`runs/20260516-134227-phase1_baseline/`) that scores 5.55 / 0.52 F1 against the published 44.34 / 4.67 BioGEN 2025 baseline. The eval module is independently confirmed correct: re-scoring the official `baseline_output.json` against `baseline_labels.json` reproduces 44.34 exactly. The 38.79 pp / 4.15 pp gap is therefore methodological, not algorithmic. Concretely:

* Pipeline produced 555 distinct support PMIDs; only ~30 intersect the labels pool. The other ~525 were never shown to a human assessor and the BioGEN macro counts them as false positives.
* Same shape on contradict: 569 emitted, ~3 in the pool.

The `baseline_labels.json` pool was built from the organizers' baseline picks only. Standard TREC pool bias — any retrospective system that didn't contribute to the original 2025 pool sees its novel-but-correct PMIDs scored as wrong.

Hardware constraints carried over from Phase 1: WSL2 with 12 GB RAM, Quadro T1000 with 4 GB VRAM, 37 GB Lucene BM25 index on native ext4. Sequential model loading remains the rule. OSS-first project policy makes Llama-3.1-70B via Together.ai a natural default for the LLM judge, with OpenAI as a paid alternative when latency or accuracy demand it.

## Goals / Non-Goals

**Goals:**
- Separate methodological from algorithmic improvements. Methodology comes first because every ablation downstream depends on it being trustworthy.
- Make every Phase 2 number directly comparable to Phase 1 and to the published 44.34 / 4.67 anchor. Dual-pool reporting (official + expanded) is mandatory for every variant.
- Keep ablations cheap. The three Vector 2 variants each must run within one local-overnight budget so we can iterate quickly.
- Hit at least one positive result on the expanded pool — a single variant that out-scores Phase 1 with honest measurement is enough to disprove the "everything was pool bias" null hypothesis.
- Stay OSS-first by default, but allow paid backends behind a config switch for cost/latency-bound work.
- Preserve the Phase 1 contract: `_maybe_run()` resume mode, sequential model loading, on-disk Parquet hand-off between phases, `task_a_output.json` shape unchanged so the official validator still passes.

**Non-Goals:**
- NLI fine-tuning on SciFact / HealthVer / BioNLI — Phase 3 once we know what's actually limiting us.
- Query rewriting agents, agentic retrieval judges, or any LLM in the live retrieval critical path — Phase 4.
- Submission to TREC BioGEN 2026. The natural way to fully escape pool bias is participating in the next round, but that's a calendar problem and out of scope here.
- Full-corpus dense encoding. 26.8 M docs at 768 dims is ~80 GB and infeasible on a 4 GB GPU. The hybrid variant deliberately encodes only a 5 M-doc subset.
- Re-running the BM25 index build. Phase 1's `data/indexes/pubmed_bm25/` (37 GB, segments_1 committed) is the immutable input.

## Decisions

### D1 — LLM-judge first, ablations after

Vector 1 (LLM-judge re-judgement) lands before any Vector 2-4 variant runs to completion. Justification: without honest pool-expanded numbers, every ablation result is uninterpretable — we can't tell whether a variant scored 6.0 or 3.0 on the official pool because it's actually better, worse, or just differently-biased. We will not optimise blind.

**Alternatives considered:**
* *Run ablations on official pool only, defer LLM-judge.* Rejected — same trap as Phase 1: numbers move around but we can't attribute the movement.
* *Run LLM-judge once at end, not validate.* Rejected — we have 588 human-labeled triples; using them as a held-out validation set is essentially free and gives a defensible concordance number for the paper.

### D2 — Default LLM-judge backend: Together.ai + Llama-3.1-70B-Instruct

OSS-first policy from Phase 1 carries over. Llama-3.1-70B-Instruct hosted by Together.ai is open-weights, ~$0.90/M tokens (input + output combined for the 70B model), and consistently within 3-5 pp of GPT-4o on MedNLI-style classification. Paid backends (`gpt-4o-mini`, `gpt-4o`) are wired in behind the same abstraction for cost/latency-bound runs or when the OSS model fails concordance validation.

The judge module exposes a single `Judge.classify(answer_sentence, pmid, abstract) -> {Supports, Contradicts, Neutral, Not relevant}` method; backend is selected via Hydra (`judge.backend=together|openai|openai-mini`).

**Alternatives considered:**
* *Local Llama-3.1-70B.* Rejected — needs >40 GB VRAM at fp16; only fp4-quantised on CPU is feasible here and that's ~5 min per call → 100+ h for our scope.
* *GPT-4o by default.* Rejected — proprietary; ~3× more expensive than Together.ai's Llama at comparable accuracy on this task.
* *Anthropic Claude Sonnet 4.* Excellent at biomedical NLI but priced like GPT-4o; kept as a fallback option in the backend abstraction.

### D3 — Concordance validation gate

The judge must reach **≥85% per-class weighted F1** against the 588 human-labeled triples in `data/qrels/biogen2025_taskA_qrels.jsonl` before being used on novel PMIDs. Below 85%, the LLM-judge run aborts with a clear error and a numerical breakdown by class. This is the methodological equivalent of the §6.5 calibration gate from Phase 1.

**Alternatives considered:**
* *80%.* Too weak — published LLM-judge studies on biomedical NLI report 78-88% routinely; landing at 80% gives no headroom for distinguishing improvement from noise.
* *90%.* Likely unattainable without GPT-4o + careful prompt engineering; would block progress unnecessarily.

### D4 — Expanded qrels schema = official qrels schema

`biogen2025_taskA_qrels_expanded.jsonl` uses the exact same JSONL shape as `biogen2025_taskA_qrels.jsonl`: one record per `(qa_id, sentence_id, pmid, class, relevance)`. Source attribution (human vs LLM) and confidence go into optional fields (`source: "human"|"llm-together-llama-3.1-70b"|...`, `confidence: 0..1`) so existing eval code reads expanded qrels without code change.

The default behaviour of `eval/metrics.py --qrels-pool=expanded` is to union human + LLM positives. A `--source` filter can restrict to human-only for reproducing the §6.5 anchor at any time.

**Alternatives considered:**
* *Separate file format for expanded.* Rejected — doubles the parsing surface and breaks `eval/metrics.py` compatibility.
* *Inline labels in the submission file.* Rejected — that's what `baseline_labels.json` does and it's exactly what caused our pool bias problem; we want qrels separable from any submission.

### D5 — Six ablation variants, three Hydra config files apiece

Each variant lives at `configs/run/phase2_<name>.yaml` and inherits from `configs/run/phase1_baseline.yaml` via Hydra's defaults list, overriding only what changes. The six variants:

| Name | Config delta | Phase 1 reference |
|---|---|---|
| `phase2_no_rerank` | `rerank: null` (skip MedCPT-CE entirely) | gap analysis §1 |
| `phase2_no_negex` | `nli.contradict.negex: false` | gap analysis §2 |
| `phase2_allow_existing` | `selection.exclude_existing: false` | gap analysis sanity check |
| `phase2_scifive_large` | `nli/contradict: scifive_large` (fp16, chunk=4) | gap analysis §3 |
| `phase2_bm25_rm3` | `retrieval: bm25_rm3` | proposal Vector 4a |
| `phase2_hybrid` | `retrieval: hybrid_rrf` (needs encoded FAISS index ready) | proposal Vector 4b |

Variants compose: `phase2_scifive_large_no_negex` is a valid Hydra invocation. We will run the compositions only after evaluating each variant in isolation.

### D6 — Hybrid retrieval uses a 5M-doc subset, not the full corpus

Phase 1 explicitly deferred dense retrieval because full-corpus encoding is infeasible on 4 GB VRAM (~80 GB FAISS index, weeks of encoding). For Phase 2 we encode only the **top 5M abstracts by citation frequency** (using Pyserini's `--storeDocvectors` term stats as a proxy for citation count). At ~768 dims fp32 this is ~15 GB on disk and ~24 h CPU encoding overnight with `MedCPT-Article-Encoder` (`ncbi/MedCPT-Article-Encoder`, 109 M params, runs comfortably on CPU at small batches).

The hybrid path retrieves k=1000 from BM25 + k=1000 from FAISS, fuses via Reciprocal Rank Fusion (RRF, k=60), and feeds the fused top-1000 to the downstream contradict pipeline. The dense path is purely additive: documents outside the 5M subset still flow through the BM25 path.

**Alternatives considered:**
* *Full corpus dense.* Infeasible (see above).
* *Cross-encoder rerank on hybrid output.* Already proven harmful (pool bias from MedCPT-CE in Phase 1). Skip.
* *FAISS GPU.* Marginal speedup at retrieval time, doesn't help with the encoding bottleneck.

### D7 — Progress and cost instrumentation as first-class metadata

`metadata.yaml` per run gains four new fields:
* `wall_clock_seconds_per_phase`: dict — debugging-grade timing per pipeline phase.
* `vram_peak_gb_per_phase`: dict — captured via `torch.cuda.max_memory_allocated()` resets between phases.
* `judge_cost_usd`: float — total $ spent on LLM-judge calls, if any.
* `judge_token_breakdown`: `{input_tokens, output_tokens, cache_hit_rate}`.

Plus `tqdm` progress bars wrap the inner loops in `phases.segment_abstracts` (135k unique pmids) and `nli.negation.filter_negated` (~1.9M sentences). Phase 1's silent 28-min waits in those phases were the worst debugging experience.

### D8 — Resume mode is preserved and extended

Phase 1's `BIOGEN_RUN_DIR=<path>` env var + `_maybe_run()` skip-if-output-exists pattern remains. We extend it for Phase 2 so ablation variants can **share intermediate parquets cheaply**: e.g., `phase2_no_negex` can reuse `retrieval_contradict.parquet` and `segmented_contradict.parquet` from a previous Phase 1 run, only recomputing the NLI pass. A new flag `--reuse-from=<run_dir>` copies/symlinks the upstream artefacts into the new run dir before invocation.

### D9 — Per-variant report row is the unit of reporting

`reports/phase2_summary.md` is regenerated after every run. One row per variant, columns:

| variant | F1@official-pool (Sup/Con) | F1@expanded-pool (Sup/Con) | Δ official→expanded | wall-clock | VRAM-peak | LLM-judge $ |

The Δ column is the headline: it isolates how much of a variant's score comes from genuine architectural improvement vs from how its picks happen to overlap (or not) with the pool.

### D10 — Backend abstraction enables paper-friendly comparison

The judge `Backend` interface has three concrete implementations (`TogetherLlama70B`, `OpenAIMini`, `OpenAI4o`) and one parameter (`max_concurrent`). Switching backend for a re-judgement run requires only `+judge.backend=openai` on the CLI. This unlocks an honest "judge sensitivity" experiment for the paper: re-judge a 200-pair sample with all three backends, report concordance pairwise. If concordance is high, the published F1 numbers are robust to judge choice; if low, that's itself a methodological contribution worth reporting.

## Risks / Trade-offs

- **[LLM-judge agreement below 85%]** → Backend abstraction lets us escalate to GPT-4o on the validation set quickly; if even GPT-4o doesn't reach 85%, that's a hard limit on this methodology and we'd fall back to a stricter "judge agreement floor" reporting (e.g., only include LLM judgements where two backends agree).
- **[Together.ai rate limits or downtime]** → Failover to OpenAI is one config flag; cost accounting in `metadata.yaml` makes the trade-off transparent.
- **[SciFive-large 30-h runs hit a transient failure mid-way]** → Resume mode (D8) means we restart at the failed phase, not from scratch. NegEx + segmentation parquets are cheap to re-cache.
- **[Hybrid retrieval encoding takes longer than 24 h on CPU]** → Encoding is one-off; we can let it run for a weekend. Acceptable.
- **[Expanded pool grows so large that F1 macro stabilises near zero for every variant]** → Cap expanded-pool size per cell (e.g., top-50 LLM-judged); the dual-pool report still tells us the relative ranking even if absolute numbers shift.
- **[Resume mode encourages stale artefact reuse]** → `metadata.yaml` records the git SHA of every contributing run; the report flags variants whose intermediates predate the variant's own config to prevent silent staleness.
- **[Judge concordance high on the easy 588-triple validation set but lower on novel out-of-pool triples]** → Document this explicitly in `reports/llm_judge_validation.md`; treat the 85% gate as necessary but not sufficient; sample 50 LLM judgements manually for spot review per backend.
- **[Phase 2 doesn't move the needle on the expanded pool]** → That would be a genuine null result and reportable. The paper still has the pool-bias methodology contribution, which stands on its own.

## Migration Plan

This is an additive change. No Phase 1 artefacts are removed or renamed. Roll-out:

1. Land the engineering cross-cuts (D7: `tqdm`, cost/VRAM instrumentation, `--reuse-from`) so they're available to every subsequent variant.
2. Land the LLM-judge module + concordance validation gate (D1, D2, D3). Run validation against the 588-triple pool, achieve ≥85%, commit the validation report.
3. Generate expanded qrels using the LLM-judge on the ~1100 novel PMIDs from Phase 1's pipeline output (D4).
4. Score the existing Phase 1 outputs on both pools — establishes the Phase 2 starting line.
5. Run the three cheap ablations (D5: `no-rerank`, `no-negex`, `allow-existing`) one by one. Update `phase2_summary.md` after each.
6. Run `scifive_large_contradict` (D5). 30 h wall-clock, overnight + a day.
7. Build the hybrid retrieval prerequisites: encode the 5M-doc FAISS subset (D6). One-off overnight.
8. Run `bm25_rm3` and `hybrid_rrf` (D5).
9. Final consolidated `reports/phase2_summary.md` + `reports/llm_judge_validation.md`. Tag `phase2-baseline`.

Rollback: each variant is its own run dir; deleting it has no effect on Phase 1 artefacts. The new code (LLM-judge module, configs, evaluation flag) is opt-in; the default `python -m trec_biogen.pipeline.run_task_a` invocation produces an unchanged Phase 1-shape run.

## Open Questions

* **Should the expanded pool include partial labels?** The 10-question CSV sample shipped by Deepak Gupta uses `partially supports` / `Partial-Supported`; the main labels file does not. If we expand the pool ourselves, do we honour partial as a separate class (and what does the LLM-judge prompt look like for it)? Resolving this affects the Relaxed vs Strict split, which is currently degenerate (identical).
* **Cap on LLM-judge calls per run?** Defaulting to "judge everything missing" is honest but unbounded for the deeper hybrid variants. A reasonable cap is "top-30 BM25 per cell per class" (~6000 calls max), which still bounds cost at <$10.
* **How to handle TREC pool bias in the paper write-up?** Frame as a methodological contribution (LLM-judge as pool expander, validated against humans) or as a caveat to algorithmic results? Probably both, structured as a separate section.
* **Do we attempt a Phase 2.5 ensemble after Vectors 2-4 finish?** I.e., combine the best support-side variant with the best contradict-side variant. Out of scope for this change but plausibly a 1-day add-on.
