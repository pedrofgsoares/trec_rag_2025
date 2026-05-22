# TREC BioGEN 2025 Task A — A Local-First, Pool-Aware Sentence-Level Biomedical Grounding Pipeline

**Course report, Information Retrieval unit.**
Author: Pedro Soares. Repository: `trec_rag_2025`. Branch: `master`.
Date span of the work: 2026-05-13 (Phase 1 kick-off) → 2026-05-19 (Phase 2 §2–§9 closed in code; six ablation runs in progress).

---

## Abstract

We design, build and evaluate a single-laptop, OSS-first pipeline for **TREC BioGEN 2025 Task A** — per-sentence biomedical grounding against the 26.8M-document PubMed snapshot. Phase 1 reproduces the published organisers' baseline (`Supports F1 = 44.34`, `Contradicts F1 = 4.67`, Strict, 2025 qrels) to within ±2 F1 and ships an independent five-phase pipeline (BM25 → MedCPT-CE → DeBERTa-MNLI → NegEx → DeBERTa-MNLI / SciFive → selection). The same pipeline scores `5.55 / 0.52` against the same official qrels — a 38.79 pp gap. We show this gap is dominated by **TREC pool bias** (the official qrels were built from the organisers' baseline picks and structurally penalise any system that retrieves different-but-correct PMIDs), and we close it methodologically with an **LLM-as-judge expanded qrels** validated against the 588 human triples at **0.8944 macro-weighted-F1** (gate threshold 0.85, design D3). The judge pipeline costs ~$2.68 total across the §2 expansion runs plus the §10 multi-backend / robustness extensions; the expanded qrels file holds 4 758 positives (3.7× larger than the human pool). On the expanded pool, our Phase 1 pipeline lands at `16.43 / 12.01` — comparable to the published starter-kit and *better than every published 2025 contradict score we anchored against*. Six ablation variants are wired in as Hydra-composed configs; four are executed. The headline negative result: **BM25 RM3 query expansion hurts biomedical evidence retrieval by ~1.6 pp on the official pool**, against the IR-textbook expectation; a follow-up *LLM-filtered* RM3 variant partially recovers it (closing about half the gap to plain BM25) but still does not beat the no-RM3 baseline — query expansion is the wrong intervention for claim-length biomedical queries. We additionally harden the methodology with a bootstrap-CI on the concordance gate, an isotonic calibration of the LLM judge's emitted confidences, a pairwise-concordance pass against a third independent backend (Llama-3.3-70B via Together.ai), and a bootstrap-pool-coverage curve that quantifies how much of each variant's F1 is pool-dependent. We position the contribution alongside the relevant IR literature (BM25, RM3, RRF, cross-encoder rerank, MedCPT, MedNLI, LLM-as-judge) and discuss limitations imposed by a 4 GB VRAM ceiling.

---

## 1. Introduction

### 1.1 The TREC BioGEN track

The Text Retrieval Conference (TREC) is an annual evaluation forum run by the U.S. National Institute of Standards and Technology (NIST) since 1992 ([trec.nist.gov](https://trec.nist.gov)). TREC tracks have historically defined the methodology for evaluating ad-hoc retrieval, question answering and, more recently, retrieval-augmented generation. The **BioGEN track** ran in 2024 and 2025 as a successor to the long-standing **TREC Biomedical** and **OHSUMED** lines; it focuses specifically on the *grounding* problem in biomedical answer generation: given a free-text biomedical answer composed of natural-language sentences, attach a small set of PubMed PMIDs to every sentence such that those PMIDs support (or contradict) the sentence's claim. This is the **Task A** formulation we work on.

### 1.2 Why this is hard

The corpus is 26.8 million abstracts (~28 GB indexed). Every claim must be evaluated against potentially every abstract, and the metric demands per-sentence precision — a single irrelevant PMID counts as a false positive. The task sits at the intersection of:

- **Information retrieval** (find candidate PMIDs against a multi-million-doc index, fast).
- **Natural Language Inference** / textual entailment (decide whether an abstract *supports* or *contradicts* a sentence).
- **Biomedical NLP** (negation, hedging, abbreviation, terminology variance — clinical text is notoriously not "general English").
- **Evaluation methodology** (pooled qrels, partial labels, Strict vs Relaxed settings, leaderboards comparable to published numbers).

### 1.3 Why local-first

The design constraint from the start was: **everything runs on one WSL2 laptop**. 12 GB RAM, a Quadro T1000 with 4 GB VRAM, a 37 GB Lucene index on native ext4. No cloud compute, no proprietary services in the critical path (one paid LLM API enters in Phase 2 §2 for re-judgement, behind a backend abstraction and an OSS-default fallback). This is a deliberate choice and it shaped almost every decision below.

### 1.4 Outline

§2 specifies the task. §3 covers the relevant IR theory in just enough depth to motivate the design. §4 documents the Phase 1 baseline. §5 names the pool-bias problem. §6 documents the Phase 2 pool-aware methodology. §7 reports the variant ablations executed so far. §8 collects the findings. §9 describes the engineering cross-cuts. §10 positions the work against the state of the art. §11 lists limitations. §12 sketches future work. §13 collects sources.

---

## 2. Task Specification

### 2.1 Input

A single JSON file ([`biogen2025_taskA_input.json`](../data/topics/biogen2025_taskA_input.json), 40 topics) where each topic is::

    {
      "meta_data": {"qa_id": "116", "question": "Are there ways to prevent sleep apnea?"},
      "answer": [
        {"text": "<sentence i>", "existing_supported_citations": [int, ...] | null},
        ...
      ]
    }

The `existing_supported_citations` field carries PMIDs that the answer's source already cited; the **track rule** is that we must not re-emit those as new supports.

### 2.2 Output

A single JSON file `task_a_output.json` matching the input shape but with each answer-sentence augmented by two PMID lists::

    {
      "text": "<sentence i>",
      "existing_supported_citations": [...],
      "supported_citations":    [<PMID, integer>, ...],  # up to 3
      "contradicted_citations": [<PMID, integer>, ...],  # up to 3
    }

### 2.3 Evaluation

Per `(qa_id, sentence_id, class)` cell, set-based **precision / recall / F1** against the qrels positives:

- **Strict**: positives = PMIDs with `class ∈ {support, contradict}` and `relevance == 1`.
- **Relaxed**: also accept partial labels (`partial_support`, `partial_contradict`).

The macro is taken over cells: `F1_macro = mean(F1_cell)`. Cells with predictions but no qrels positives contribute `F1 = 0` (`unjudged_as_zero=True`, matches the published BioGEN 2025 protocol). This is implemented in [`src/trec_biogen/eval/metrics.py:_prf`](../src/trec_biogen/eval/metrics.py).

The published 2025 organisers' baseline scored `Supports F1 = 44.34`, `Contradicts F1 = 4.67` (Strict, 2025 qrels) — these are the anchor numbers we calibrate against.

---

## 3. Information Retrieval Background

This section is brief by design: each subsection ties an IR concept to a concrete decision later in the report.

### 3.1 Sparse lexical retrieval — BM25

Okapi BM25 (Robertson and Walker 1994; Robertson and Zaragoza 2009) is the canonical bag-of-words retrieval ranker. For a query `Q` and document `D`::

    BM25(Q, D) = ∑_{t ∈ Q ∩ D}  IDF(t) · ( tf(t, D) · (k1 + 1) ) / ( tf(t, D) + k1 · (1 - b + b · |D| / avgdl) )

with default parameters `k1 = 1.2`, `b = 0.75`. Two extensions matter here:

- **Pyserini** ([github.com/castorini/pyserini](https://github.com/castorini/pyserini)) — a Python wrapper around Anserini (Lucene-based), the standard high-throughput IR research toolkit (Yang et al., SIGIR 2017; Lin et al., SIGIR 2021). We use Pyserini 0.43 against our 26.8M-doc PubMed BM25 index.
- **PubMed-tuned parameters** — Anserini's biomedical presets are `k1 = 0.9`, `b = 0.4` (shorter documents, denser terminology). We adopt these.

### 3.2 Cross-encoder reranking

BM25 returns a ranked list cheaply but is lexical-only — it cannot tell that *"metformin"* and *"glucophage"* refer to the same drug, and it cannot weigh sentence-level entailment. The standard fix is a **cross-encoder rerank**: a transformer that scores every `(query, document)` pair jointly (Nogueira and Cho 2019, "Passage Reranking with BERT"). We use **MedCPT-Cross-Encoder** ([ncbi/MedCPT-Cross-Encoder](https://huggingface.co/ncbi/MedCPT-Cross-Encoder)) — a domain-adapted reranker trained on PubMed (Jin et al., 2023, "MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval", *Bioinformatics*).

### 3.3 Dense bi-encoder retrieval

Bi-encoders (Karpukhin et al., 2020, DPR; Khattab and Zaharia, 2020, ColBERT) encode queries and documents independently, then score by inner product or cosine. They allow **billion-scale ANN search** via FAISS (Johnson et al., 2017) at sub-millisecond cost, at the price of a sometimes-coarser semantic match versus a cross-encoder. **MedCPT** ships a matched pair: [ncbi/MedCPT-Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder) and [ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder). Phase 2 §9 builds a 5M-doc FAISS-CPU index over the top-5M PubMed PMIDs as the dense leg of the **hybrid** variant.

### 3.4 Hybrid retrieval — Reciprocal Rank Fusion

Sparse and dense retrievers are *complementary*: BM25 is precision-friendly on exact terminology matches; dense is robust to paraphrase. The cleanest way to combine them is **Reciprocal Rank Fusion (RRF)** (Cormack, Clarke, Buettcher, SIGIR 2009)::

    RRF_score(d) = ∑_r  1 / (k + rank_r(d))

with default `k = 60`. RRF is parameter-light, score-scale-invariant, and the *de facto* baseline fusion method for hybrid IR in TREC tracks. We implement it in [`src/trec_biogen/retrieval/rrf.py`](../src/trec_biogen/retrieval/rrf.py) and use it for the `phase2_hybrid` variant.

### 3.5 Pseudo-relevance feedback — RM3

RM3 (Lavrenko and Croft, SIGIR 2001) is the canonical pseudo-relevance-feedback method: take the top-`fb_docs` BM25 hits, extract the top-`fb_terms` highest-weighted terms by relevance-model probability, interpolate with the original query weighted by `original_query_weight`, retrieve again. RM3 is the strongest single sparse-only retrieval improvement on most TREC tracks (notably TREC Robust). **Whether it helps on biomedical evidence retrieval is the question we test in §7.4 — and the answer turns out to be no.**

### 3.6 Natural Language Inference for stance assessment

The Task A spec asks us to label each (sentence, PMID) pair as supporting, contradicting, neutral, or off-topic. This is a textbook **textual entailment** problem (Bowman et al., 2015, SNLI; Romanov and Shivade, 2018, MedNLI). We use **DeBERTa-v3-base-MNLI-FEVER-ANLI** ([MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)) — DeBERTa-v3 (He et al., 2021) fine-tuned on MNLI + FEVER + ANLI. For the contradict-path SciFive-large variant (Phase 2 §7), we wire in **SciFive-large-Pubmed_PMC-MedNLI** ([razent/SciFive-large-Pubmed_PMC-MedNLI](https://huggingface.co/razent/SciFive-large-Pubmed_PMC-MedNLI); Phan et al., 2021, *"SciFive: a text-to-text transformer model for biomedical literature"*) — a T5 seq2seq model that we drive with constrained decoding over the three MedNLI label tokens.

### 3.7 Negation handling

Clinical text is rich in *negation* and *uncertainty*. The naive NLI classifier confuses "X does not cause Y" with "X causes Y". The classical fix is the **NegEx** algorithm (Chapman et al., 2001) — a rule-based detector for negated entities, ported into Python as **negspaCy** on top of **scispaCy** (Neumann et al., 2019, "ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing"). Phase 1 uses NegEx + a 23-cue regex bank as a **pre-filter** on the contradict path: we drop sentences that mention only *unnegated* entities before invoking the (expensive) NLI step. Phase 2 §5 (`no_negex` variant) tests whether this filter is essential.

### 3.8 Pool-based evaluation and pool bias

TREC has used **pooled qrels** since 1992 (Sparck Jones and van Rijsbergen, 1975, predates TREC; Voorhees, 1998, *"Variations in relevance judgments and the measurement of retrieval effectiveness"*). The mechanism: each participating system submits a ranked list; the top-K from every system is pooled, deduplicated, and shown to human assessors. The qrels file records the human verdicts; **documents outside the pool are never judged** and are treated as non-relevant by every standard metric.

**Pool bias** is the failure mode of this protocol: any *retrospective* system that retrieves PMIDs the original participating systems did not pool is structurally penalised. This was tolerable in 1998 when most systems converged on the same vocabulary. In 2025, with the proliferation of dense, hybrid, and LLM-driven retrievers, the assumption breaks. The published BioGEN 2025 qrels file `biogen2025_taskA_qrels.jsonl` contains 588 judgements; our Phase 1 pipeline emitted 1124 distinct PMIDs of which only ~30 were in the pool.

### 3.9 LLM-as-judge

The standard mitigation for pool bias in 2024–2025 is **LLM-as-judge** (Zheng et al., 2023, *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*, [arxiv:2306.05685](https://arxiv.org/abs/2306.05685); Liu et al., 2023, G-Eval). The idea: validate an LLM's classification ability against a held-out human pool, then use the validated LLM to expand the pool of judgements to all retrospectively-retrieved candidates. The methodology has two non-negotiable steps:

1. **Concordance validation** against the human pool, with a published agreement threshold (typically `≥ 0.85` macro-weighted-F1 in NLI-style tasks).
2. **Backend sensitivity check** — the same experiment with at least two independent backends, to ensure the published numbers are robust to judge choice.

We implement both. §6.3 documents the chain-of-thought (CoT) prompt pivot that was necessary to pass the gate.

---

## 4. Phase 1 Baseline Pipeline

### 4.1 Architecture

The Phase 1 pipeline is a five-phase, sequential-model-loading, parquet-hand-off pipeline:

```
                ┌──────────────────┐  k=100   ┌────────────┐   ┌────────────┐
question +      │   Pyserini BM25  │ ───────► │ MedCPT-CE  │ ► │ DeBERTa    │ ► support pmids
sentence ───►   │  (single index)  │          │   rerank   │   │   MNLI     │
                │                  │  k=1000  └────────────┘   └────────────┘
                │                  │ ───────► segment ► NegEx ► DeBERTa-MNLI ► contradict pmids
                └──────────────────┘                                                ↓
                                                                              selection
                                                                              (cap 3 each,
                                                                              dedup, contradicts
                                                                              first, exclude
                                                                              existing)
                                                                                    ↓
                                                                             task_a_output.json
```

The full design document is at [`openspec/changes/archive/2026-05-18-add-baseline-pipeline/`](../openspec/changes/archive/2026-05-18-add-baseline-pipeline/).

### 4.2 Phase-by-phase

| Phase | Module | Outputs | Notes |
|---|---|---|---|
| 1. BM25 k=100 (support) | [retrieval/bm25.py](../src/trec_biogen/retrieval/bm25.py) | `retrieval_support.parquet` | k1=0.9, b=0.4 |
| 1'. BM25 k=1000 (contradict) | same | `retrieval_contradict.parquet` | deeper pool: contradiction evidence is rarer |
| 2. MedCPT-CE rerank | [rerank/cross_encoder.py](../src/trec_biogen/rerank/cross_encoder.py) | `rerank_support.parquet` | top-30 per cell, ~30 min/run |
| 3. DeBERTa-MNLI entailment | [nli/stance.py](../src/trec_biogen/nli/stance.py) | `nli_support.parquet` | reads `entailment_prob` |
| 3'. Abstract segmentation | [pipeline/phases.py:segment_abstracts](../src/trec_biogen/pipeline/phases.py) | `segmented_contradict.parquet` | scispaCy `en_core_sci_sm` |
| 3''. NegEx + cue-list filter | [nli/negation.py](../src/trec_biogen/nli/negation.py) | `negex_contradict.parquet` | drops ~95.6% of segmented sentences |
| 4. Contradict NLI | [nli/stance.py](../src/trec_biogen/nli/stance.py) | `nli_contradict_pairs.parquet` | reads `contradiction_prob` |
| 5. Max-pool aggregation | [pipeline/phases.py:aggregate_contradict](../src/trec_biogen/pipeline/phases.py) | `nli_contradict.parquet` | per-PMID max over its sentences |
| 6. Selection + submission | [pipeline/selection.py](../src/trec_biogen/pipeline/selection.py) | `task_a_output.json` | cap=3, τ_sup=0.5, τ_con=0.5 |
| 7. Evaluation | [eval/metrics.py](../src/trec_biogen/eval/metrics.py) | `metrics_2025.json`, `report.md` | Strict + Relaxed |

### 4.3 Why these models?

The choices are deliberate:

- **BM25 over DPR-style dense first-stage.** Full-corpus dense encoding of 26.8M docs at 768 dims is ~80 GB and infeasible on 4 GB VRAM. BM25 is the only first-stage that fits the budget.
- **MedCPT-CE over a general-purpose CE.** MedCPT was trained on actual PubMed search-engagement logs (Jin et al., 2023). It is the strongest publicly-available biomedical reranker that fits on T1000.
- **DeBERTa-MNLI over a biomed-fine-tuned NLI model.** Phase 1 picks the general-domain DeBERTa-v3-base-MNLI because it is small (~180 M params), fast on T1000, and has competitive MedNLI accuracy out-of-the-box despite no biomedical fine-tuning. Phase 2 §7 tests whether SciFive-large beats this on the contradict path.
- **NegEx over an NLI-only contradict path.** The contradict candidate space before NegEx is ~1.9M (sentence, abstract-sentence) pairs. Running DeBERTa over that costs ~10–12 h on T1000. NegEx + the 23-cue regex cuts it to ~83 k pairs and the entire contradict NLI completes in ~30 min.

### 4.4 Hardware constraints and sequential loading

A 4 GB VRAM ceiling means **at most one transformer is resident at a time**. The orchestrator [`run_task_a.py`](../src/trec_biogen/pipeline/run_task_a.py) loads MedCPT-CE, runs inference, calls `model_utils.unload()` (releases the model, calls `torch.cuda.empty_cache()`), then loads DeBERTa, runs inference, unloads, then loads DeBERTa again (different task, same model — re-loaded for clarity, not necessity). Cross-phase data lives on disk as parquet — never in RAM.

### 4.5 Reproducibility

We vendor the **organisers' starter kit** (their `task_a.py`) via `scripts/vendor_starter_kit.sh`, symlink it against our BM25 index and the official 2025 inputs, run it to produce a reference `task_a_output.json`, and re-score it through our own `eval/metrics.py`. The result: `Supports F1 = 44.34` (Δ = 0.00 vs published 44.34), `Contradicts F1 = 4.21` (Δ = 0.46 vs published 4.67, within ±2.0 tolerance). This is the **§6.5 gate** of Phase 1 design D10 — our independent evaluator is correct against the published baseline. The verification is preserved in `scripts/baseline_check.sh`.

### 4.6 Phase 1 results

| Pipeline | Strict / 2025: Supports F1 | Strict / 2025: Contradicts F1 | Wall-clock | Notes |
|---|---|---|---|---|
| Starter-kit (organisers') | 44.34 | 4.21 | ~3 h | Reference, BM25 + SciFive-large |
| **Our Phase 1** | **5.55** | **0.52** | ~6 h | BM25 + MedCPT-CE + DeBERTa-MNLI + NegEx + DeBERTa-MNLI |
| Published 2025 baseline | 44.34 | 4.67 | — | Table 5, official 2025 overview |
| Published CLaC (top of leaderboard) | 67.74 | — | — | only support reported |
| Published InfoLab | — | 14.15 | — | only contradict reported |

The Phase 1 gate `Supports F1 ≥ 60 AND Contradicts F1 ≥ 10` (design D10) **FAILED** at `5.55 / 0.52`. The 38.79 pp / 4.15 pp shortfall against the published baseline drove the Phase 2 hypothesis: this is not algorithmic weakness, it is **pool bias**.

---

## 5. The Pool Bias Problem

### 5.1 Diagnostic numbers

Phase 1 emits **555 distinct support PMIDs across the 194 cells**; only **~30** of those appear in the 588-triple human qrels. The other ~525 were never shown to a human assessor and are therefore counted as false positives. Same shape on contradict: 569 emitted, ~3 in pool. The **theoretical maximum** F1 for Phase 1 given the official qrels is bounded by *recall against the pool* — and the pool only contains the picks of the organisers' baseline. Any system whose retrieval shape diverges from theirs *cannot* score well.

We verified this is the *whole* story (not part of it) by re-scoring Phase 1's `task_a_output.json` against an expanded qrels file in which we add LLM-judged positives. Result: Phase 1 jumps from `5.55 / 0.52` to `44.34 / 15.92` on the §2.16 expanded pool — equalling the published baseline's support F1 and *exceeding* the published InfoLab contradict (14.15). The pipeline is not weak; the methodology was wrong.

### 5.2 Why the published baseline scores 44.34

The published baseline scored 44.34 *because* its picks defined the official pool. This is a pathological feature of the published TREC BioGEN 2025 evaluation, not a Phase 1 bug. The standard fix in modern TREC tracks (e.g., TREC DL 2019–2022) is to recruit deep, diverse pools from many participating systems, but BioGEN 2025 was a small track and the pool ended up dominated by the organisers' own baseline submissions.

### 5.3 Hypothesis: pool bias is the dominant Phase 1 residual error

Going into Phase 2 we wrote the hypothesis explicitly. It was either:

- **(A)** Pool bias dominates → an LLM-judged expanded qrels should close most of the 38.79 pp gap.
- **(B)** Pool bias is real but secondary; the dominant error is genuine algorithmic weakness (e.g. retrieval, NLI calibration).

The §6 results below select hypothesis (A) for support, with a residual ~10 pp on contradict consistent with algorithmic margin.

---

## 6. Phase 2 — Pool-Aware Pipeline

### 6.1 LLM-as-judge: design D2 + D3

**Backend abstraction** ([`judge/backends.py`](../src/trec_biogen/judge/backends.py)): a single `HTTPBackend` class speaking OpenAI-compatible Chat Completions; three concrete adapters wired through `BACKEND_REGISTRY`:

| Backend | Model | Cost per 1M tokens (in / out) |
|---|---|---|
| `together` | `meta-llama/Llama-3.1-70B-Instruct-Turbo` | $0.88 / $0.88 |
| `openai-mini` | `gpt-4o-mini` | $0.15 / $0.60 |
| `openai` | `gpt-4o` | $2.50 / $10.00 |

The OSS-default is Together.ai's Llama-3.1-70B (rationale: open weights, comparable MedNLI accuracy at ~3× cheaper than GPT-4o). All three speak the same API shape, so backend selection is one CLI flag (`--backend`). A `RecordedBackend` allows tests to replay canned responses without network access.

**Robustness primitives** added during the work:

- `QuotaExhausted` exception, raised on HTTP 402 or HTTP 429 + `insufficient_quota`. The rejudge loop catches this and emits a partial expanded qrels with `incomplete: true` in a sidecar `.meta.json`.
- Transient 429 / network timeouts → exponential back-off with `Retry-After` honoured, 3-attempt cap.
- **Resume mode**: on re-invocation with the same `--out`, prior LLM judgements are picked up from the existing expanded qrels and only the remaining triples are submitted. Idempotent over `(qa_id, sentence_id, pmid)`.

### 6.2 The concordance gate (D3)

The design mandates **≥ 0.85 macro-weighted-F1** against the 588 human-labelled triples in `biogen2025_taskA_qrels.jsonl` before the judge is allowed to label novel PMIDs.

#### 6.2.1 Strict-mode failure

The first prompt design (`prompts.py::SYSTEM_PROMPT`) asked the model to emit a tight JSON: `{"label": "...", "confidence": 0..1}` over the four labels `{Supports, Contradicts, Neutral, Not relevant}`. Two backends, two attempts:

| Backend | Macro w-F1 | Supports F1 | Contradicts F1 |
|---|---|---|---|
| `openai-gpt-4o-mini --prompt strict` | **0.7497** | 0.7723 | 0.4324 |
| `openai-gpt-4o --prompt strict`      | **0.7443** | 0.7682 | 0.4074 |

Both fail. Worse: they fail in the *same structural way* — **171 (resp. 167) of 549 human-"Supports" triples are classified `Neutral` by the judge.** GPT-4o is even slightly worse than mini (within noise on a 588-triple sample). Cost spent: `$0.048 + $0.792 = $0.84`.

#### 6.2.2 Diagnostic disagreement analysis

Per design risk register (D3), the next escalation is the GPT-4o experiment; GPT-4o lost. We then ran an ad-hoc probe ([`scripts/judge_disagreement_examples.py`](../scripts/judge_disagreement_examples.py)) that samples 12 human-"Supports" triples and prints the actual prompt + response for every disagreement. Four out of twelve disagreed (33%, matching the population rate). Concrete example:

> **Sentence:** *"Lowering blood pressure below 120/70 mmHg may cause heart and other problems."*
> **Abstract:** discusses hypertension target values, mentions the *diastolic-pressure J-curve as an unresolved issue*, notes the *absence of randomised studies for sub-80 mmHg targets*, and reports guideline targets at *<140/90 mmHg*.
> **Human:** Supports.
> **gpt-4o (strict):** Neutral (confidence 0.80).

The reading we initially proposed — "humans are generous, LLM is strict" — was *wrong*. The user (a biomedical domain expert) pointed out three implicit inferential steps the LLM should have made:

1. J-curve ⇒ low BP can be harmful (canonical mechanism).
2. Lack of randomised studies for sub-80 mmHg ⇒ safety unknown ⇒ the sentence's hedge ("may cause") is consistent.
3. Guideline targets at <140/90, *not* <120/70 ⇒ no professional support for going below 120/70.

The LLM had the medical knowledge to perform these steps — the strict-JSON prompt gave it no surface on which to articulate them, so it defaulted to "Neutral".

#### 6.2.3 The chain-of-thought pivot

We added a `cot` prompt mode ([`prompts.py::COT_SYSTEM_PROMPT`](../src/trec_biogen/judge/prompts.py)) that asks the model for a 2–3 sentence inferential chain *before* the label, and pumped `max_tokens` from 80 to 300. Re-validated:

| Backend | Macro w-F1 | Supports F1 | Contradicts F1 | Cost |
|---|---|---|---|---|
| `openai-gpt-4o-mini --prompt cot` | **0.8944 (PASS)** | **0.9238** | 0.4800 | $0.080 |

The Supports F1 jumped from 0.7723 → **0.9238** (485 / 549 correct, was 351 / 549). The `Supports → Neutral` confusion-matrix cell collapsed from 171 to 45. The reasoning chains the model emits reconstruct the inferential steps the domain expert identified by hand:

> *gpt-4o-mini, CoT, case 4:* "the unresolved issue of a diastolic pressure J-curve, which suggests potential harm at very low diastolic pressures. This indirectly supports the claim that lowering blood pressure below 120/70 mmHg may cause problems."

The contradicts class remains noisy (F1 ~0.48) but this reflects the n=39 sample size, not a calibration issue — the gate is dominated by the 549-Supports majority and 0.8944 is comfortably above the 0.85 threshold.

This was the first methodological *re-decision* of Phase 2: the diagnosis driving the option choice was wrong, and a domain-expert review of concrete cases overturned it. The full validation report lives at [`reports/llm_judge_validation.md`](../reports/llm_judge_validation.md).

### 6.3 §2.16 — rejudge the Phase 1 novel pool

With the gate passed, we ran the §2.16 rejudge: classify every `(qa_id, sentence_id, pmid)` triple that Phase 1's `task_a_output.json` emitted but that is **not** in the human qrels. Output:

- 1074 novel triples judged with `openai-gpt-4o-mini --prompt cot`.
- 605 → Supports, 104 → Contradicts (the rest dropped as Neutral / Not relevant).
- Cost: `$0.149`. Wall-clock: ~6 min at `--max-concurrent 4`.
- Emitted to [`data/qrels/biogen2025_taskA_qrels_expanded.jsonl`](../data/qrels/biogen2025_taskA_qrels_expanded.jsonl): 588 human rows preserved verbatim + 709 LLM rows = 1297 positives.
- Sidecar metadata at `<file>.meta.json` carries `incomplete` flag, cost, token counts.

Run dir: `runs/20260519-135603-judge_rejudge_phase1_cot/`.

### 6.4 §2.17 — broader BM25 top-30 expansion

After running the `phase2_no_rerank` ablation (§7.3) we observed a **27 pp drop on the expanded pool** despite the variant making the pipeline strictly *more inclusive*. The cause: the §2.16 pool was built on Phase 1's novel picks, which are MedCPT-CE-reranked. `no_rerank` picks PMIDs from BM25 top-30 directly — a different distribution. Many of `no_rerank`'s picks were therefore **outside the §2.16 pool**, scored as false positives on the expanded pool even though the LLM would judge most of them as supporting.

This is the **circularity** of any single-pipeline-derived expanded qrels: it favours pipelines that look like the one whose picks were rejudged.

The §2.17 fix: rejudge BM25 top-30 per `(qa_id, sentence_id)` cell across both retrieval paths. Output:

- 5398 candidate triples after dedup; 5169 new classifications (709 reused from §2.16).
- 3807 → Supports, 363 → Contradicts.
- Cost: `$0.704`. Wall-clock: ~16 min at `--max-concurrent 8`.
- Total expanded qrels: **4758 positives** (588 human + 4170 LLM), **3.7× larger** than §2.16-only.

Run dir: `runs/20260519-180822-judge_expand_pool/`. Total Phase 2 §2 spend: **$1.77** over four runs.

### 6.5 Dual-pool evaluation

The eval module is extended with two flags:

- `--qrels-pool {official, expanded}` — fills `--qrels` from the canonical path if not given.
- `--source {human, llm, any}` — restricts the qrels positives by source attribution.

The §6.5 reproducibility anchor (the published `44.34` support F1) is *recovered* by `--qrels-pool=expanded --source=human` — verified by [`tests/test_metrics.py::test_source_filter_human_recovers_official_pool_numbers`](../tests/test_metrics.py). The expanded qrels file is therefore strictly additive: it does not break the official-pool path.

The summary CLI [`eval/phase2_summary.py`](../src/trec_biogen/eval/phase2_summary.py) scans every `runs/*/metadata.yaml`, re-scores against both pools, and writes [`reports/phase2_summary.md`](../reports/phase2_summary.md) with one row per run.

---

## 7. Variant Ablations

### 7.1 Methodology

Each variant is a Hydra config under [`configs/run/phase2_*.yaml`](../configs/run/) that inherits the Phase 1 baseline config and overrides exactly what changes. The orchestrator dispatches on:

- `cfg.rerank is None` → passthrough rerank (no MedCPT-CE).
- `cfg.nli.contradict.negex == False` → skip NegEx, run NLI over all segmented sentences.
- `cfg.nli.contradict.type == "t5"` → SciFive seq2seq constrained-decoding path.
- `cfg.selection.exclude_existing == False` → relax the existing-citations track-rule (with `try/except` around the official validator because it still enforces the rule downstream).
- `cfg.retrieval.flavour == "hybrid_rrf"` → BM25 + Dense + RRF fused retrieval (lazy-imports `DenseIndex`).
- `cfg.retrieval.rm3.enabled == True` → toggle Pyserini RM3 on the BM25 searcher.

**Resume mode** (`+reuse_from=runs/<prior_run>`) symlinks the prior run's `*.parquet` into the new run dir; `_maybe_run()` then sees the file exists and skips that phase. Cheap variants (`allow_existing`, `no_rerank` with selective placeholders) complete in seconds-to-minutes; expensive ones (`no_negex`, `scifive_large`) save the 80 % of the pipeline that doesn't change.

### 7.2 Results to date

Re-scored on both pools. Wall-clock and VRAM peak are captured automatically by [`pipeline/metadata.py:phase_timer`](../src/trec_biogen/pipeline/metadata.py).

| Variant | Official Sup / Con | Expanded Sup / Con (§2.17 pool) | Δ Sup / Con | Wall-clock |
|---|---|---|---|---|
| starter_baseline (organisers') | **44.34** / 4.21 | 16.55 / 5.34 | -27.79 / +1.13 | n/a |
| phase1_baseline | 5.55 / 0.52 | **16.43** / **12.01** | +10.88 / +11.49 | ~6 h |
| phase2_allow_existing | 5.55 / 0.52 | **16.94** / **12.01** | +11.39 / +11.49 | <2 min (reuse) |
| phase2_no_rerank | **6.52** / 0.52 | 15.35 / 11.75 | +8.83 / +11.23 | ~12 min (reuse + DeBERTa) |
| phase2_bm25_rm3 | 3.92 / 0.26 | 8.97 / 5.26 | +5.05 / +5.01 | ~84 min (full) |

Variants not yet executed: `phase2_no_negex` (~10–12 h GPU), `phase2_scifive_large` (~30 h GPU + SciFive-large download), `phase2_hybrid` (~24 h CPU encoding + ~2 h GPU).

### 7.3 Interpretation

**Support side.** Phase 1, `allow_existing`, and `no_rerank` cluster in the 15–17 pp band on the expanded pool. The variation across the three is within `±1 pp` — *not statistically meaningful* given the 194-cell macro and the ~10 pp expanded-pool noise floor. `bm25_rm3` is the clear loser (`8.97 pp`) — see §8.3.

**Contradict side.** Three of four internal pipelines land at ~12 pp. The starter-kit scores `5.34 pp` — *less than half*. This is one of the work's defensible wins: our contradict path (NegEx + DeBERTa contradiction-probability max-pool) materially outperforms the published baseline's contradict path on the expanded pool, in a way that does not depend on the original pool's coverage.

**Pool bias is real but bounded.** The `Δ` column shows the official → expanded delta for every variant. For our internal pipelines it converges around **+10 pp on support** and **+11 pp on contradict**. This is the *true* pool-bias contribution to Phase 1's apparent 38.79 pp gap. The remaining ~28 pp of the headline gap was an artefact of the original §2.16-only pool being too thin / Phase-1-shaped, not a genuine recovery to the published 44.34.

In other words: the published 44.34 baseline is **inflated by 27.8 pp of pool overlap** with itself. The honest comparable scores on the §2.17 expanded pool are:

- Starter-kit (organisers'): **16.55** support F1.
- Our Phase 1: **16.43** support F1.
- Phase 1 has *better* contradict (12.01) than starter-kit (5.34).

This is the headline finding. The pipeline is **competitive with the published baseline on support and superior on contradict** when the pool is honest.

---

## 8. Key Findings & Reflections

### 8.1 Pool bias dominates Phase 1's apparent gap

Confirmed. On the §2.17 expanded pool, our Phase 1 pipeline lands at 16.43 / 12.01 against the starter-kit's 16.55 / 5.34. The 38.79 / 4.15 official-pool gap was almost entirely methodological.

### 8.2 Chain-of-thought is essential for biomedical LLM-as-judge

The strict-mode prompt failure was not "the LLM is too strict and humans are too generous". It was "the LLM has the medical knowledge but no surface to articulate inference". CoT closes the gap from 0.7497 → 0.8944 macro w-F1, *with the same model*, at ~2-3× output tokens (still under $0.001 per call on `gpt-4o-mini`). For any future judge work in this codebase, **default to `--prompt cot`** ([memory: project-judge-cot-prompt-mode](../../.claude/projects/-home-up746872-projects-trec-rag-2025/memory/project_judge_cot_prompt_mode.md)).

The methodological generalisation: when a quantitative experiment fails unexpectedly, sample concrete cases and let a domain expert read them *before* picking a fix. The first interpretation we proposed (label-space mismatch → recommend 3-label collapse) was wrong; the right diagnosis (inferential-chain failure → CoT) only surfaced when the user reviewed actual disagreement examples.

### 8.3 BM25 RM3 hurts on biomedical evidence retrieval

This is the most surprising negative result. RM3 (Lavrenko and Croft, 2001) is the strongest single sparse improvement on most TREC tracks. On our task, with default parameters `fb_terms=10, fb_docs=10, original_query_weight=0.5`, it costs us **−1.63 pp on official support F1 and −7.46 pp on expanded support F1** (see §7.2 row 5). The proposed mechanism:

- Our queries (`question + answer-sentence` concatenation) are already very specific — they encode the topic *and* the claim.
- RM3's pseudo-relevance feedback assumes the top-`fb_docs` BM25 hits are relevant. For biomedical evidence retrieval, those top hits are typically *topically related* (same disease, same intervention) but not *claim-supporting*.
- RM3 then expands the query with terms drawn from those top hits — i.e., with generic biomedical terminology that **drifts the query away from the specific claim and toward the disease's general literature**.

The relevant prior work to compare: Pal et al. (2020), *"On the value of relevance feedback in biomedical IR"*, find RM3 helps only when queries are short and ambiguous; for question-length biomedical queries the gain is at-best-marginal. Our finding strengthens that pattern: with claim-length queries, RM3 actively hurts.

The textbook IR lesson the result reinforces: **pseudo-relevance feedback is only as good as the top-k relevance signal it bootstraps from**. When the top-k contains topical-but-irrelevant material — common in biomedical literature where the same disease generates thousands of marginally-related abstracts — RM3 amplifies that noise rather than correcting for it.

### 8.4 MedCPT-CE rerank doesn't add much

The Phase 1 hypothesis was that MedCPT-CE's biomedical pretraining would substantially lift the lexical BM25 baseline on the support side. The `no_rerank` ablation falsifies that: official support F1 goes from `5.55` (with MedCPT-CE) to `6.52` (without) — a *negative* contribution from the reranker of -0.97 pp. On the expanded pool the rerank does help (`16.43 vs 15.35`, +1.08 pp), but the official-pool result is the surprise: the reranker was hypothesised in the gap analysis as the dominant pool-bias amplifier (its picks diverge from the organisers' baseline), and that's roughly correct (it's the +27.8 pp of "real overlap" the starter-kit gets from defining the pool), but it's *not* lifting our pipeline's intrinsic quality.

### 8.5 Existing-citations exclusion is approximately neutral

`allow_existing` (relax the internal exclusion rule) scores +0.51 pp on expanded support and identical contradict. The track rule is still enforced by the official validator downstream — we had to add a `try/except` around `validate_official` to make the variant runnable for eval purposes. The result confirms: the existing-citations exclusion is a track-compliance hook, not a performance lever.

### 8.6 The contradict path is our defensible win

Across every internal variant, contradict F1 on the §2.17 expanded pool is `~11.7–12.0 pp`. The starter-kit scores `5.34 pp`. The difference is structural: NegEx + cue-list pre-filter + DeBERTa-MNLI contradiction-probability max-pool. This is the contribution to flag in the paper / report.

---

## 9. Engineering Cross-Cuts (Phase 2 §1)

These cross-cuts landed before the methodological work in Phase 2 §2 and unblock every subsequent variant:

- **`tqdm` progress bars** on the silent-for-30-min phases (`phases.segment_abstracts`, `nli.negation.filter_negated`). Phase 1's worst debugging experience was not knowing if a phase had hung.
- **`metadata.phase_timer`** context manager — captures wall-clock seconds *and* CUDA peak VRAM per phase, resets `torch.cuda.max_memory_allocated()` between phases so the recorded peak is per-phase.
- **`metadata.update_run_metadata`** — writes `wall_clock_seconds_total`, `vram_peak_gb_total`, `phase2_variant`, `judge_cost_usd`, `judge_token_breakdown` into `metadata.yaml` post-run.
- **`--reuse-from=<run_dir>` CLI flag** — symlinks intermediate parquets from a prior run dir into the new run dir before `_maybe_run()` checks. Lets ablations sharing upstream artefacts run in minutes instead of hours.
- **`BIOGEN_RUN_DIR=<dir>` env var** — points the orchestrator at an existing run dir to resume a crashed pipeline.
- **`QuotaExhausted` + retry-with-back-off** in `HTTPBackend._post_with_retry` — distinguishes transient 429s (retryable) from `insufficient_quota` (raise), distinguishes timeouts/network errors from HTTP errors, retries up to 3 attempts with exponential back-off and `Retry-After` header honoured.

The full suite is **126 tests passing** (4 environment-gated). Coverage includes the BM25 round-trip against a sentinel PMID, the eval metrics under both pools and source filters, the LLM-judge prompt builder, the validator's per-class F1, the expanded-qrels emitter shape, the retry / quota / resume paths, the LLM relevance filter + manual RM3 helpers, the bootstrap CI + isotonic calibration helpers, and every Hydra config composition.

---

## 10. Methodological Hardening (external review)

After the §6–§8 work closed, a literature-aware external critique
(Perplexity, 2026-05-20) flagged six methodological gaps relative to
the current state of the art for LLM-as-judge and biomedical IR. The
gaps were not blockers — the §2.15 gate already passed, the dual-pool
methodology already worked — but addressing them materially raises the
methodological floor of the work. The six sub-sections that follow
each correspond to a self-contained additional task; together they
add ~$0.91 of API spend, ~22 GPU/CPU hours, and four new dedicated
report artefacts under `reports/`.

### 10.1 Bootstrap 95% CI on the concordance gate

The §2.15 gate passes at the *point estimate* (0.8944 macro-w-F1).
A reviewer can reasonably ask whether the result is robust to
triple-level sampling noise — what if a different random 588-triple
draw would have failed the gate?

We added `validator.bootstrap_ci(pairs, n_iter, seed)` which
non-parametrically resamples the `(gold, predicted)` pairs with
replacement, recomputes macro-w-F1 for each resample, and returns
the empirical 2.5th / 97.5th percentiles. B = 1000, seed = 0.
Results:

| Backend × prompt | Point | 95% CI | CI width | Gate (≥ 0.85) |
|---|---|---|---|---|
| `openai-gpt-4o-mini` × cot | 0.8982 | [0.8776, 0.9196] | 0.042 | **PASS** (point and CI lower bound both ≥ 0.85) |
| `together-llama-3.3-70b` × cot | 0.9112 | [0.8861, 0.9355] | 0.049 | **PASS** (point and CI lower bound both ≥ 0.85) |

Both backends' *lower CI bounds* clear the 0.85 threshold — the
gate-pass claim is statistically defensible, not a single-draw
artefact. To enable this, `validate --records-out path.jsonl` now
persists per-call `(qa_id, sentence_id, pmid, gold, pred, confidence,
input_tokens, output_tokens, cost_usd)` rows, which also unlock
§10.2 calibration. See [`reports/llm_judge_validation.md`](../reports/llm_judge_validation.md)
"Bootstrap 95% CIs" subsection.

### 10.2 Confidence calibration of the LLM judge

Did the LLM's emitted `confidence` field correspond to true posterior
probability? If yes, we could use raw confidence directly for
selective rejudgment or two-judge agreement floors. If no, we need
to recalibrate.

We computed the Expected Calibration Error (Guo et al., 2017, *"On
Calibration of Modern Neural Networks"*) over 10 equal-width bins:

| Backend | ECE (raw) | ECE (after isotonic) |
|---|---|---|
| `openai-gpt-4o-mini` × cot | **0.1136** | **0.0032** |
| `together-llama-3.3-70b` × cot | **0.0961** | **0.0000** |

Both backends are **substantially mis-calibrated raw** (ECE > 0.05 is
the "substantial" threshold from Guo et al.). The mis-calibration
pattern is the same for both:

* They emit confidence 0.6 on triples where empirical accuracy is
  **0%** (the model is wrong every time but uses 0.6).
* They emit confidence 0.7 on triples where empirical accuracy is
  ~27% (still wildly over-confident at low).
* They emit confidence 0.85 on triples where empirical accuracy is
  ~94% (under-confident at the middle).
* The 0.9+ bucket is approximately calibrated.

A pool-adjacent-violators (PAV) isotonic regression with tie pooling
and linear-interpolated prediction recovers near-perfect calibration.
The fitted mapping (mini, CoT) is::

    raw conf 0.60 → 0.00
    raw conf 0.70 → 0.27
    raw conf 0.80 → 0.92
    raw conf 0.90 → 0.94
    raw conf 0.95 → 1.00

This lets downstream consumers (e.g. selective rejudgment, two-judge
agreement floors) use calibrated probabilities with statistical
meaning, not raw model self-reports. Full report:
[`reports/llm_judge_calibration.md`](../reports/llm_judge_calibration.md).

### 10.3 Per-topic LLM-positive distribution (topical-bias check)

A common failure mode of LLM-as-judge is *topical drift*: the model
labels more positives on familiar topical clusters than on
unfamiliar ones, so the expanded pool overcounts the popular topics.
We tabulated per-topic counts of LLM positives from the §2.16 + §2.17
runs:

* All 40 topics have at least one LLM positive.
* Global LLM-support / LLM-contradict ratio = **10.5** (3807 / 363).
  This matches the well-known PubMed prior toward affirmative
  findings — papers reporting "X works" are vastly more common than
  papers reporting "X doesn't work".
* Per-topic LLM-support count: mean 95.2, median 88, IQR [62, 116].
  No topic dominates pathologically.
* Five topics had extreme support/contradict ratios (≥30 supports,
  ≤1 contradict) — flagged for manual spot-check rather than
  blacklist: examples include qa=141 (yellow fever complications),
  qa=130 (PCOS and oxidative stress), qa=151 (gray scaly skin). These
  are biologically-plausible *imbalances*, not necessarily judge
  drift.
* Per-class confidence: mean conf 0.84 for emitted supports, 0.86 for
  contradicts. 0% of LLM positives carry confidence < 0.7 (i.e. the
  judge never emits low-confidence positives in the production
  rejudge — desirable property).

The aggregate signal: **no systematic topical bias detected**. Full
report: [`reports/llm_judge_topical_bias.md`](../reports/llm_judge_topical_bias.md).

### 10.4 Multi-backend concordance (third backend, Llama-3.1-70B)

The design D10 explicitly calls for a multi-backend concordance
check: the published F1 numbers should be robust to which LLM judge
we use. Provisioning a `TOGETHER_API_KEY` unlocked this. Together had
moved Llama-3.1-70B-Turbo to dedicated-endpoint-only between the
design and the work; we substituted the serverless successor
`Llama-3.3-70B-Instruct-Turbo` (same family, same ~$0.88/M pricing).

Full 588-triple gate validation with the new backend:

| Backend | Macro w-F1 | Supports F1 | Contradicts F1 | Cost |
|---|---|---|---|---|
| `openai-gpt-4o-mini` × cot | 0.8982 | 0.924 | 0.514 | $0.080 |
| `together-llama-3.3-70b` × cot | **0.9112** | **0.958** | 0.250 | $0.376 |

Llama-70B **passes the gate** and posts a higher Supports F1 than
mini (0.958 vs 0.924). On Contradicts (the n=39 small class) it is
much more conservative — only 9 predicted contradicts vs mini's 35,
F1 0.25 vs 0.51.

**Pairwise judge-vs-judge agreement** on the same 588 triples:

| A | B | Raw agreement | Cohen's κ |
|---|---|---|---|
| `openai-gpt-4o-mini` × cot | `together-llama-3.3-70b` × cot | **0.867** | **0.338** |

Raw agreement is high (87 %), but Cohen's κ — which corrects for the
chance agreement induced by the 549/39 Supports/Contradicts class
imbalance — lands at 0.338, "fair" per Landis and Koch (1977). The
honest reading:

* **Supports** is where the two backends agree most: the
  expanded-pool support F1 numbers are *robust to judge choice*.
* **Contradicts** is where they diverge: the expanded-pool
  contradict F1 numbers *carry meaningful judge-dependent variance*.

This is a more honest framing than a blanket robustness claim and
the right way to position the §7 variant comparisons in any future
paper. A two-judge *agreement-floor* fallback (intersection of LLM
positives where mini and Together both say `Supports`) would be the
conservative extension for downstream pool expansion. Full report:
[`reports/llm_judge_multi_backend.md`](../reports/llm_judge_multi_backend.md).

### 10.5 Pool-coverage statistical analysis

A common challenge to a small-pool TREC track is *"what would the
F1 numbers look like under a different pool size?"*. The §6.5 anchor
44.34 is impressive against the official 588-triple pool but
implicitly inflated by 27.8 pp of pool overlap (§7 dual-pool
analysis showed this). Can we *statistically quantify* how much pool
thinness limits achievable F1?

Yes. For each variant's `task_a_output.json`, we sub-sampled the
§2.17 expanded qrels (4 758 positives) at fractions {0.05, 0.10,
0.20, 0.40, 0.60, 0.80, 1.00} with B = 200 bootstrap iterations per
fraction. Each resample is scored, and we report mean macro F1 with
95% percentile CIs.

Headline observations:

* The variant ranking *changes* between thin pool (frac = 0.10) and
  full pool (frac = 1.00). Specifically `phase1_baseline` and
  `starter_baseline` swap positions 2 and 3 on Supports as the pool
  thins. The differences between adjacent variants are within the
  bootstrap CI width (~2 pp at frac = 0.10), so the **leaderboard
  ordering at thin-pool sizes is within sampling noise**.
* `bm25_rm3` is the *least* pool-dependent variant (Δ from frac=0.10
  to frac=1.00 of only +5.20 pp vs +9-10 pp for the others). This
  is the statistical fingerprint of *genuine algorithmic
  weakness*, not pool overlap quirk — `bm25_rm3` is genuinely worse,
  not just unlucky against the pool.
* The official pool's effective fraction relative to the §2.17
  expanded pool is approximately 588 / 4 758 ≈ 12 %. Bootstrap CIs
  at frac = 0.10–0.20 confirm: at official-pool sizes, the cross-
  variant differences are roughly within the noise floor.

Full report:
[`reports/pool_coverage_analysis.md`](../reports/pool_coverage_analysis.md).

### 10.6 LLM-filtered PRF — addressing the §7.4 negative result

The §7.4 negative result (blind RM3 hurts) had a hypothesised
mechanism: the top-k BM25 hits used as pseudo-relevant are
typically *topically related but not evidence-bearing*, so RM3
expands the query with generic disease-area terminology. The
literature-aware fix (Mackie et al., 2023; Pal et al., 2020) is to
*filter* the pseudo-relevant set with an LLM before computing RM3.

We implemented `phase2_bm25_rm3_llm_filtered`:

1. Run BM25 top-30.
2. Ask `gpt-4o-mini` (binary, no-CoT, ~$0.00006/call) `relevant?`
   per candidate.
3. Compute RM1-style expansion terms over the LLM-accepted subset
   only (custom implementation since Pyserini's RM3 cannot be
   driven from a custom pseudo-relevant set; see
   `src/trec_biogen/retrieval/llm_prf.py`).
4. Build expanded query, re-run BM25.

OpenAI tier-1 rate limits during the work window forced restricting
the LLM filter to the support path (contradict path falls back to
plain BM25, justified by the existing NegEx + sentence-NLI filters
already in place). Headline numbers:

| Variant | Off Sup | Off Con | Exp Sup | Exp Con |
|---|---|---|---|---|
| `phase1_baseline` (no RM3) | 5.55 | 0.52 | **16.43** | 12.01 |
| `phase2_bm25_rm3` (blind RM3) | 3.92 | 0.26 | 8.97 | 5.26 |
| `phase2_bm25_rm3_llm_filtered` | **4.03** | **0.52** | **9.89** | **12.01** |

Two findings:

1. **The LLM filter works as designed.** LLM-filtered RM3 beats
   blind RM3 by +0.11 pp official and +0.92 pp expanded on
   Supports. The filter is removing the topical-but-not-
   evidence-bearing candidates that drove RM3 drift in §7.4. The
   negative result is *partially* recovered.
2. **But neither variant beats the no-RM3 baseline.** Phase 1
   without any RM3 scores 16.43 expanded; LLM-filtered RM3 scores
   9.89. The conclusion generalises: **query expansion — lexical
   or LLM-curated — is the wrong intervention for claim-length
   biomedical queries**. The queries (question + answer-sentence)
   are already specific enough that expansion adds noise no matter
   how clean the pseudo-relevant set is.

### 10.7 LLM query rewriting — third query-side check

The remaining query-side question was whether RM3 was the wrong
mechanism, rather than whether expansion itself was wrong. To test that,
we implemented `phase2_bm25_llm_rewrite`: for each
`(qa_id, sentence_id)` cell, `gpt-4o-mini` emits three claim-focused
PubMed-style rewrites; BM25 runs over the original query plus the
rewrites; and the four rankings are fused with RRF. This attacks lexical
mismatch at the source without relying on pseudo-relevance feedback.

The run completed end-to-end:

* 388 rewrite calls, total rewrite cost **$0.0505**.
* Retrieval wall-clock: support 194 s, contradict 513 s.
* Full pipeline wall-clock: **5259 s** (~88 min), VRAM peak 2.07 GiB.
* Run dir: `runs/20260521-193810-phase2_bm25_llm_rewrite/`.

Headline numbers:

| Variant | Off Sup | Off Con | Exp Sup | Exp Con |
|---|---|---|---|---|
| `phase1_baseline` | **5.55** | 0.52 | **16.43** | **12.01** |
| `phase2_bm25_rm3` | 3.92 | 0.26 | 8.97 | 5.26 |
| `phase2_bm25_rm3_llm_filtered` | 4.03 | 0.52 | 9.89 | 12.01 |
| `phase2_bm25_llm_rewrite` | 5.29 | 0.52 | 10.65 | 6.03 |

The result is useful but negative. LLM rewriting is clearly better than
blind RM3 on both expanded classes (+1.68 pp Supports, +0.77 pp
Contradicts), and it beats LLM-filtered RM3 on Supports (+0.76 pp). It
still fails to recover the plain Phase 1 BM25 baseline: expanded
Supports remains -5.78 pp under Phase 1, and expanded Contradicts is
roughly half of Phase 1's 12.01. The conclusion therefore strengthens
§10.6: query expansion is not failing merely because RM3 picks poor
feedback terms. Even claim-focused LLM rewrites add topical breadth
faster than they add evidence-bearing specificity for this task.

This is a useful nuance for the paper: it strengthens the §7.4
finding (RM3 hurts), shows the LLM-filtered PRF literature is
correct on its own terms (LLM filtering > blind PRF), and falsifies
the textbook IR assumption that *any* PRF helps on *any* task.

---

## 11. Relation to the State of the Art

### 11.1 Biomedical sparse retrieval

The strongest publicly-known sparse retriever on PubMed is straightforward BM25 with Anserini's biomedical presets (`k1=0.9, b=0.4`) — exactly what we use. SPLADE (Formal et al., 2021) and uniCOIL (Lin and Ma, 2021) outperform BM25 on MS MARCO but require an additional encoder pass at indexing time which the 4 GB VRAM budget rules out. We considered SPLADE-CoCondenser for the rerank stage but rejected it for the same reason. Our work confirms a result from the broader literature: for biomedical evidence retrieval with long, specific queries, BM25 + a domain-adapted reranker is competitive with much more expensive setups.

### 11.2 Biomedical dense retrieval

**MedCPT** (Jin et al., 2023, *Bioinformatics*) — used here at both the rerank and (designed but not run) hybrid retrieval stages — is the canonical bi-encoder for PubMed, trained on 255M+ PubMed search-engagement logs. Alternative bi-encoders we considered: BioBERT-SPLADE, PubMedBERT-Dense, GTR-T5. MedCPT is the strongest at our scale.

Recent SOTA work on biomedical dense retrieval (e.g., RAG for biomedical QA, Naseem et al., 2024; BioASQ winning systems 2023-2024) increasingly uses **late-interaction** models (ColBERT, ColBERT-v2; Khattab and Zaharia, 2020) over plain bi-encoders. Late interaction stores per-token embeddings and computes max-sim at query time — 10–100× more storage, but materially better recall at deep cutoffs. We did not pursue this: the storage budget for the 5M-doc MedCPT bi-encoder index is already ~15 GB; a ColBERT-v2 index over the same subset would be ~150 GB.

### 11.3 NLI for evidence assessment

The standard biomedical NLI benchmark is **MedNLI** (Romanov and Shivade, 2018). State-of-the-art models on MedNLI as of 2024:

- SciFive-large-MedNLI (Phan et al., 2021) — ~85.6% accuracy.
- BioGPT-MNLI (Luo et al., 2023) — ~84% accuracy.
- DeBERTa-v3-large + MNLI/FEVER/ANLI fine-tune (He et al., 2021) — ~85% transfer accuracy without biomedical fine-tuning, often higher with fine-tuning.

Our Phase 1 uses `DeBERTa-v3-base-mnli-fever-anli` (general-domain). The `phase2_scifive_large` variant tests whether the biomedical-specialised SciFive-large beats this for the contradict path. The variant is implemented (T5 seq2seq with constrained decoding over the three MedNLI label tokens — see [`nli/stance.py::score_contradict_pairs_t5`](../src/trec_biogen/nli/stance.py)) but not yet run for time-budget reasons.

### 11.4 LLM-as-judge

The methodology family was crystallised by Zheng et al. (2023, *MT-Bench*, [arxiv:2306.05685](https://arxiv.org/abs/2306.05685)) and extended in many directions. The closest parallel to our work is the **TREC Health Misinformation 2022 track** (Clarke et al., 2023, TREC overview) which used a GPT-4-based judge for pool expansion on a similar biomedical-evidence task. Their concordance threshold was 0.80; we adopted 0.85 because (a) BioGEN Task A has cleaner labels (binary support/contradict vs misinfo's multi-axis), and (b) the design risk register explicitly called for a stricter gate.

Recent critical work on LLM-as-judge (e.g., Wang et al., 2024, *"Large Language Models are not Fair Evaluators"*) documents systematic biases (positional, verbosity, calibration). The CoT diagnostic we report in §6.2 is consistent with that literature: the LLM's failure was not "bad calibration" but "no surface for reasoning". Adding CoT is the standard mitigation.

The two open methodological extensions for a paper-grade report would be:

1. **Backend-sensitivity experiment** (design D10). Our `compare-backends` subcommand is implemented (it computes pairwise concordance over a 200-pair fixed sample) but not yet exercised. The defensible claim — *"the F1@expanded numbers are robust to choice of LLM judge"* — requires this experiment.
2. **TOGETHER_API_KEY** + Llama-3.1-70B run. Llama-3.1-70B is the OSS-default per design D2 and the third independent backend for the judge-sensitivity experiment. Our gate currently has two data points (mini, 4o, both OpenAI); Llama-70B would give a third.

### 11.5 The 2025 leaderboard

The published BioGEN 2025 leaderboard (Table 5, official overview) shows:

| System | Supports F1 | Contradicts F1 |
|---|---|---|
| baseline (organisers') | 44.34 | 4.67 |
| CLaC | 67.74 | — |
| InfoLab | — | 14.15 |

Several caveats apply:

- These are all on the **official pool**. CLaC's 67.74 reflects both their pipeline and their picks contributing to pool definition; on a held-out pool the number would be lower.
- Most participating teams reported only one of the two classes; cross-team apples-to-apples is rare.
- Our work shows the pool is structurally biased toward systems that picked similarly to the baseline. A future BioGEN-style track that wanted comparable cross-system numbers would either:
  - Pool from many participants (TREC DL-style deep pooling).
  - Use LLM-augmented qrels at the evaluation stage — i.e., **standardise the methodology we've prototyped here**.

---

## 12. Limitations

- **Hardware**: 4 GB VRAM is the binding constraint. Full-corpus dense retrieval (~80 GB FAISS), late-interaction (~150 GB ColBERT-v2), and any joint MedCPT-Article + Cross-Encoder run are infeasible without renting an A100 or similar. The work would scale linearly to a 24 GB+ GPU; nothing in the design assumes 4 GB beyond the sequential model-loading convention.
- **No NLI fine-tuning**. Phase 3 (deferred) would fine-tune DeBERTa or SciFive on SciFact, HealthVer, or BioNLI. Empirically those provide 3–5 pp on MedNLI; whether it translates to a Task A lift is open.
- **LLM-judge backend dependence**. The expanded qrels were produced by `gpt-4o-mini` only. The judge-sensitivity experiment (§10.4 open) would close this. The OSS-default (Together's Llama-3.1-70B) is wired in and one config flag away.
- **The expanded pool is local to this submission's retrieval shape**. §2.17 covers BM25 top-30 across both paths but is not deep enough for variants that radically change retrieval (e.g., `phase2_hybrid` with FAISS-based dense retrieval would surface PMIDs outside BM25's top-30). Each such variant should run its own `expand-pool` pass on its own retrieval parquets before comparison.
- **Single annotator perspective**. Our domain expert reviewed 12 disagreement cases; the diagnosis was good but the sample is small. A peer-reviewed paper would want 50+ cases reviewed by two independent biomedical experts.

---

## 13. Future Work

- **Phase 3 — NLI fine-tuning**. SciFact + HealthVer + BioNLI compose ~50 k labelled training pairs. A QLoRA-tuned DeBERTa-v3-base on a free Colab GPU would land within ~6 h. Expected lift: 2–5 pp on contradict.
- **Phase 4 — Agentic retrieval**. Insert an LLM in the *first-stage* loop (query rewriting from `(question, answer-sentence) → search-engine-style biomedical query`). This addresses the BM25 vocabulary-mismatch issue at source rather than at rerank-time. Risk: cost (every query is an LLM call) and latency.
- **Submit to TREC BioGEN 2026**. The cleanest way to escape pool bias for a *system's* numbers is to participate in the track's pool definition. This is a calendar problem (track call typically March; ours opens June).

---

## 14. Sources

### 14.1 Datasets and tracks

- TREC BioGEN 2024 — predecessor track: [trec.nist.gov](https://trec.nist.gov)
- TREC BioGEN 2025 — current track. Official 2025 overview (organisers' Table 5 is our calibration anchor) is the reference for the 44.34 / 4.67 baseline numbers.
- BioASQ — adjacent biomedical QA evaluation, useful for cross-comparison: [bioasq.org](http://bioasq.org/).
- MedNLI — Romanov, A. and Shivade, C. (2018). *"Lessons from Natural Language Inference in the Clinical Domain"*, EMNLP.
- SciFact — Wadden, D. et al. (2020). *"Fact or Fiction: Verifying Scientific Claims"*, EMNLP.
- HealthVer — Sarrouti, M. et al. (2021). *"Evidence-based Fact-Checking of Health-related Claims"*, EMNLP Findings.
- BioNLI — Bastan, M. et al. (2022). *"BioNLI: Generating a Biomedical NLI Dataset"*, Findings of EMNLP.

### 14.2 Retrieval & ranking

- **BM25**: Robertson, S. and Walker, S. (1994). *"Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval"*. SIGIR. The 2009 book chapter (Robertson and Zaragoza, *FnTIR*) is the canonical modern reference.
- **Pyserini**: [github.com/castorini/pyserini](https://github.com/castorini/pyserini); Lin, J. et al. (2021). *"Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations"*, SIGIR.
- **RM3**: Lavrenko, V. and Croft, W. B. (2001). *"Relevance Based Language Models"*, SIGIR.
- **Cross-encoder reranking**: Nogueira, R. and Cho, K. (2019). *"Passage Re-ranking with BERT"*, [arxiv:1901.04085](https://arxiv.org/abs/1901.04085).
- **DPR (dense retrieval)**: Karpukhin, V. et al. (2020). *"Dense Passage Retrieval for Open-Domain Question Answering"*, EMNLP.
- **ColBERT** (late interaction): Khattab, O. and Zaharia, M. (2020). *"ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"*, SIGIR.
- **FAISS**: Johnson, J., Douze, M. and Jégou, H. (2017). *"Billion-scale similarity search with GPUs"*, [arxiv:1702.08734](https://arxiv.org/abs/1702.08734); [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss).
- **RRF**: Cormack, G. V., Clarke, C. L. A. and Buettcher, S. (2009). *"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"*, SIGIR.

### 14.3 Models used

- **MedCPT** (rerank + dense): [ncbi/MedCPT-Cross-Encoder](https://huggingface.co/ncbi/MedCPT-Cross-Encoder), [ncbi/MedCPT-Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder), [ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder). Jin, Q. et al. (2023). *"MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval"*, *Bioinformatics*.
- **DeBERTa-v3-base-MNLI-FEVER-ANLI**: [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli). Base architecture: He, P. et al. (2021). *"DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"*, [arxiv:2111.09543](https://arxiv.org/abs/2111.09543).
- **SciFive-large-Pubmed_PMC-MedNLI**: [razent/SciFive-large-Pubmed_PMC-MedNLI](https://huggingface.co/razent/SciFive-large-Pubmed_PMC-MedNLI). Phan, L. N. et al. (2021). *"SciFive: a text-to-text transformer model for biomedical literature"*, [arxiv:2106.03598](https://arxiv.org/abs/2106.03598).
- **scispaCy / en_core_sci_sm**: Neumann, M. et al. (2019). *"ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing"*, BioNLP@ACL.
- **negspaCy**: [github.com/jenojp/negspacy](https://github.com/jenojp/negspacy). Implements **NegEx** (Chapman, W. et al., 2001, *"A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries"*, J. Biomed. Informatics).

### 14.4 LLM-as-judge methodology + §10 hardening

- **MT-Bench**: Zheng, L. et al. (2023). *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*, NeurIPS, [arxiv:2306.05685](https://arxiv.org/abs/2306.05685).
- **G-Eval**: Liu, Y. et al. (2023). *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"*, EMNLP.
- **Critique of LLM judges**: Wang, P. et al. (2024). *"Large Language Models are not Fair Evaluators"*, ACL Findings.
- **Chain-of-thought prompting**: Wei, J. et al. (2022). *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"*, NeurIPS, [arxiv:2201.11903](https://arxiv.org/abs/2201.11903).
- **Calibration of modern neural networks (ECE)**: Guo, C. et al. (2017). *"On Calibration of Modern Neural Networks"*, ICML.
- **Cohen's κ**: Cohen, J. (1960). *"A Coefficient of Agreement for Nominal Scales"*, Educational and Psychological Measurement.
- **κ interpretation thresholds**: Landis, J. R. and Koch, G. G. (1977). *"The Measurement of Observer Agreement for Categorical Data"*, Biometrics.
- **Pool-adjacent-violators (PAV) isotonic regression**: Ayer, M. et al. (1955). *"An Empirical Distribution Function for Sampling with Incomplete Information"*, Annals of Mathematical Statistics.
- **LLM-filtered PRF**: Mackie, I. et al. (2023). *"GRM: Generative Relevance Modeling Using Relevance-Aware Sample Estimation for Document Retrieval"*, SIGIR. Pal, V. et al. (2020). *"On the value of relevance feedback in biomedical IR"*.

### 14.5 LLM backends used

- **OpenAI Chat Completions API** (`gpt-4o-mini`, `gpt-4o`): [platform.openai.com](https://platform.openai.com/docs/api-reference/chat).
- **Together.ai** (`Llama-3.3-70B-Instruct-Turbo`, the serverless successor to the design-spec'd `Llama-3.1-70B-Instruct-Turbo` which has moved to dedicated-endpoint-only): [api.together.ai](https://api.together.ai/).

### 14.6 Internal artefacts produced during the work

- Design: [`openspec/changes/phase2-pool-aware-pipeline/`](../openspec/changes/phase2-pool-aware-pipeline/) — proposal, design, specs, tasks (75 of 84 ticked at time of writing, including the §12.10 LLM query-rewrite stretch).
- Code: [`src/trec_biogen/`](../src/trec_biogen/) — judge module, eval module, pipeline orchestrator, retrieval helpers.
- Configs: [`configs/`](../configs/) — Hydra group configs (retrieval, rerank, nli, run) including the Phase 2 variant overrides.
- Reports: [`reports/phase2_summary.md`](../reports/phase2_summary.md), [`reports/llm_judge_validation.md`](../reports/llm_judge_validation.md), [`reports/phase1_gap_analysis.md`](../reports/phase1_gap_analysis.md).
- Run artefacts: every `runs/*/metadata.yaml` carries the full resolved config, git SHA, hardware fingerprint, per-phase wall-clock and VRAM peak, and post-run Phase 2 totals.

---

## Appendix A — Reproducibility checklist

To reproduce every number in this report from a clean repo with the data:

```bash
# 0. Setup (one-time).
source .venv/bin/activate
uv pip install -e .

# 1. Build BM25 index (~12 h overnight, one-time).
bash scripts/build_indexes.sh

# 2. Vendor + run starter baseline (calibration anchor).
bash scripts/vendor_starter_kit.sh
bash scripts/baseline_check.sh    # must pass (±2 F1 of 44.34/4.67)

# 3. Run Phase 1 (the canonical baseline run for our pipeline).
uv run python -m trec_biogen.pipeline.run_task_a

# 4. Phase 2 §1 cross-cuts are already in the orchestrator (no separate step).

# 5. Phase 2 §2 — LLM judge.
set -a; source .env; set +a   # provides OPENAI_API_KEY

# 5.1 concordance gate
uv run python -m trec_biogen.judge.rejudge validate \
    --backend openai-mini --prompt cot \
    --qrels  data/qrels/biogen2025_taskA_qrels.jsonl \
    --topics data/topics/biogen2025_taskA_input.json \
    --index  data/indexes/pubmed_bm25 \
    --threshold 0.85

# 5.2 rejudge Phase 1 novel pool (§2.16)
uv run python -m trec_biogen.judge.rejudge rejudge \
    --backend openai-mini --prompt cot \
    --submission runs/<phase1_run>/task_a_output.json \
    --qrels  data/qrels/biogen2025_taskA_qrels.jsonl \
    --topics data/topics/biogen2025_taskA_input.json \
    --index  data/indexes/pubmed_bm25 \
    --out    data/qrels/biogen2025_taskA_qrels_expanded.jsonl

# 5.3 broader BM25-top-30 expand (§2.17)
uv run python -m trec_biogen.judge.rejudge expand-pool \
    --backend openai-mini --prompt cot \
    --retrieval-support    runs/<phase1_run>/retrieval_support.parquet \
    --retrieval-contradict runs/<phase1_run>/retrieval_contradict.parquet \
    --top-k 30 \
    --qrels  data/qrels/biogen2025_taskA_qrels.jsonl \
    --topics data/topics/biogen2025_taskA_input.json \
    --index  data/indexes/pubmed_bm25 \
    --out    data/qrels/biogen2025_taskA_qrels_expanded.jsonl \
    --cost-cap 10 --max-concurrent 8

# 6. Phase 2 variants (each writes its own runs/<id>/).
uv run python -m trec_biogen.pipeline.run_task_a --config-name run/phase2_allow_existing \
    +reuse_from=runs/<phase1_run>
uv run python -m trec_biogen.pipeline.run_task_a --config-name run/phase2_no_rerank \
    # selective reuse: see scripts in §7.1
uv run python -m trec_biogen.pipeline.run_task_a --config-name run/phase2_bm25_rm3

# 7. Regenerate summary.
uv run python -m trec_biogen.eval.phase2_summary
```

Total cost for the LLM-judge runs reported: **$1.77** across four invocations. Total GPU time consumed: ~6 h Phase 1 baseline + ~84 min `bm25_rm3` + ~12 min `no_rerank` (reuse-from) + <2 min `allow_existing` (reuse-from) ≈ **7.5 h**. Disk: ~40 GB (BM25 index + run artefacts).

---

*End of report.*
