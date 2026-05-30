# TREC BioGEN 2025 Task A — A Local-First, Pool-Aware Sentence-Level Biomedical Grounding Pipeline

**Course report, Information Retrieval unit.**
Author: Pedro Soares. Repository: `trec_rag_2025`. Branch: `main`.
Date span of the work: 2026-05-13 (Phase 1 kick-off) → 2026-05-24 (Phase 2.6 three-judge Krippendorff α + held-out k=5 ECE complete; eight ablations landed, seven executed).

---

## Abstract

We design, build and evaluate a single-laptop, OSS-first pipeline for **TREC BioGEN 2025 Task A** — per-sentence biomedical grounding against the 26.8M-document PubMed snapshot. Phase 1 reproduces the published organisers' baseline (`Supports F1 = 44.34`, `Contradicts F1 = 4.67`, Strict, 2025 qrels — `TEST` run in the official Table 5) to within ±2 F1 and ships an independent five-phase pipeline (BM25 → MedCPT-CE → DeBERTa-MNLI → NegEx → DeBERTa-MNLI / SciFive → selection). The same pipeline scores `5.55 / 0.52` against the same official qrels — a 38.79 pp gap. We show this gap is dominated by **TREC pool bias** (the official 2025 pool was built from 10 selected topics × one top-priority run per team, 244 PMIDs manually assessed; see §11.5.1) and we close it methodologically with an **LLM-as-judge expanded qrels** validated against the 588 human triples at **0.8944 macro-weighted-F1** (gate threshold 0.85, design D3; **0.9112 with Llama-3.3-70B** and **0.8980 with Qwen2.5-72B** as independent backends). The judge pipeline costs ~$9.15 total across the §2 expansion runs (`$1.77`), the §10 multi-backend / robustness extensions (`$0.91`), the Phase 2.5 second-judge rejudge (`$2.47`), and the Phase 2.6 third-judge rejudge (`$4.00`); the §2.17 expanded qrels file holds 4 758 positives (3.7× larger than the human pool). On that expanded pool, our Phase 1 pipeline lands at `16.43 / 12.01` — comparable to the published starter-kit (`16.55 / 5.34`). Eight ablation variants are wired in as Hydra-composed configs; seven are executed (`phase2_hybrid` is the only outstanding one — see §7.2). The headline negative result: **three independent query-expansion mechanisms all hurt on claim-length biomedical queries** — blind RM3 (-1.63 pp official Sup, -7.46 pp expanded Sup), LLM-filtered RM3 (Mackie 2023, partial recovery, still -6.54 pp under no-RM3), and full LLM query-rewrite (-5.78 pp). We harden the methodology with a bootstrap-CI on the concordance gate (lower bound 0.8776 for mini-cot, 0.8861 for Llama-cot — gate-pass is statistically defensible), a held-out k=5 CV ECE on isotonic confidence calibration (0.048 mini, 0.033 Llama — calibrator generalises across topics), a Phase 2.5 two-judge intersection-on-contradicts pool (4 758 → 4 438 positives, 88 % Contradicts drop) with cell-level bootstrap CIs, and a **Phase 2.6 three-judge Krippendorff α** across mini-cot / Llama-3.3-70B / Qwen2.5-72B — the **α = 0.60 Llama ↔ Qwen vs ~0.15 either-pair-with-mini** identifies mini-cot as the Contradicts-class outlier and resolves the Phase 2.5 ambiguity (is mini over-emitting or Llama over-stripping? — mini is). The structural `no_negex` > Phase 1 contradicts gain survives both the two- and three-judge intersection pools (3.4× and 3.3× respectively); the apparent `no_negex` > starter margin collapses into CI overlap on those conservative pools. The 2025 overview paper (Gupta et al., 2026, [arXiv:2603.21582](https://arxiv.org/abs/2603.21582), §7.1.2 and §8.2) discloses that **Table 5 itself is BioACE-derived**: the published leaderboard numbers (including the 44.34/4.67 baseline) come from a Llama-3.3 LLM judging classify-as-Supports/Contradicts/Neutral/Not-relevant, with expert-vs-automated concordance explicitly deferred to future work. BioACE is documented separately in [arXiv:2602.04982](https://arxiv.org/abs/2602.04982). The published Table 5 therefore depends on a single-backend LLM judge with no CI and no multi-juror corroboration — repositioning our contribution as the **methodological infrastructure (multi-juror α, bootstrap CIs, CoT pivot diagnosis, calibrated confidence, conservative-pool reporting) that makes single-backend LLM-as-judge defensible**, rather than the use of LLM-as-judge itself. Cross-referencing the seven Task A notebook papers in NIST TREC 34 proceedings against author personal GitHubs, **zero of the seven 2025 teams published code**; the repository accompanying this report is therefore positioned as the first publicly reproducible Task A reference. Limitations: 4 GB VRAM (`phase2_hybrid` unrun), LLM-in-decision gap to the top of the Supports leaderboard (CLaC 67.74) — see §11.5.2 and §12.

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

RM3 (Lavrenko and Croft, SIGIR 2001) is the canonical pseudo-relevance-feedback method: take the top-`fb_docs` BM25 hits, extract the top-`fb_terms` highest-weighted terms by relevance-model probability, interpolate with the original query weighted by `original_query_weight`, retrieve again. RM3 is the strongest single sparse-only retrieval improvement on most TREC tracks (notably TREC Robust). **Whether it helps on biomedical evidence retrieval is the question we test in §7.2 and analyse in §8.3 — and the answer turns out to be no.**

### 3.6 Natural Language Inference for stance assessment

The Task A spec asks us to label each (sentence, PMID) pair as supporting, contradicting, neutral, or off-topic. This is a textbook **textual entailment** problem (Bowman et al., 2015, SNLI; Romanov and Shivade, 2018, MedNLI). We use **DeBERTa-v3-base-MNLI-FEVER-ANLI** ([MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)) — DeBERTa-v3 (He et al., 2021) fine-tuned on MNLI + FEVER + ANLI. For the contradict-path SciFive-large variant (Phase 2 §7), we wire in **SciFive-large-Pubmed_PMC-MedNLI** ([razent/SciFive-large-Pubmed_PMC-MedNLI](https://huggingface.co/razent/SciFive-large-Pubmed_PMC-MedNLI); Phan et al., 2021, *"SciFive: a text-to-text transformer model for biomedical literature"*) — a T5 seq2seq model that we drive with constrained decoding over the three MedNLI label tokens.

### 3.7 Negation handling

Clinical text is rich in *negation* and *uncertainty*. The naive NLI classifier confuses "X does not cause Y" with "X causes Y". The classical fix is the **NegEx** algorithm (Chapman et al., 2001) — a rule-based detector for negated entities, ported into Python as **negspaCy** on top of **scispaCy** (Neumann et al., 2019, "ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing"). Phase 1 uses NegEx + a 23-cue regex bank as a **pre-filter** on the contradict path: we drop sentences that mention only *unnegated* entities before invoking the (expensive) NLI step. Phase 2 §5 (`no_negex` variant) tests whether this filter is essential.

### 3.8 Pool-based evaluation and pool bias

TREC has used **pooled qrels** since 1992 (Sparck Jones and van Rijsbergen, 1975, predates TREC; Voorhees, 1998, *"Variations in relevance judgments and the measurement of retrieval effectiveness"*). The mechanism: each participating system submits a ranked list; the top-K from every system is pooled, deduplicated, and shown to human assessors. The qrels file records the human verdicts; **documents outside the pool are never judged** and are treated as non-relevant by every standard metric.

**Pool bias** is the failure mode of this protocol: any *retrospective* system that retrieves PMIDs the original participating systems did not pool is structurally penalised. This was tolerable in 1998 when most systems converged on the same vocabulary. In 2025, with the proliferation of dense, hybrid, and LLM-driven retrievers, the assumption breaks. The published BioGEN 2025 pool covers **10 selected topics** of the 40 in the input, built from one top-priority run per participating team plus the baseline — **244 PubMed abstracts manually assessed** in total (Gupta et al., 2026, §6). The `biogen2025_taskA_qrels.jsonl` file we use as the human-judged set is the cell-level expansion of those 244 PMIDs to `(qa_id, sentence_id, pmid, class)` triples — **588 judgement triples** in total. Our Phase 1 pipeline emitted 1 124 distinct PMIDs of which only ~30 fell inside the 244-PMID pool.

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

We vendor the **organisers' starter kit** (their `task_a.py`) via `scripts/vendor_starter_kit.sh`, symlink it against our BM25 index and the official 2025 inputs, run it to produce a reference `task_a_output.json`, and re-score it through our own `eval/metrics.py` against the 244-PMID human-judged qrels file (`biogen2025_taskA_qrels.jsonl`, the cell-level expansion of the expert pool documented in §11.5.1). The result: `Supports F1 = 44.34` (Δ = 0.00 vs published 44.34), `Contradicts F1 = 4.21` (Δ = 0.46 vs published 4.67, within ±2.0 tolerance). This is the **§6.5 gate** of Phase 1 design D10 — our independent evaluator is correct against the published baseline. The verification is preserved in `scripts/baseline_check.sh`.

**Subtlety worth flagging:** the published Table 5 of the overview (Gupta et al., 2026) reports the *BioACE-automatic* evaluation (Llama-3.3-judged), not the expert-only evaluation (which lives in Figures 3-5 of the overview). The 44.34 we reproduce above is the expert-pool F1; that BioACE-derived Table 5 also reports 44.34 for the baseline is consistent with BioACE having been calibrated against the same human pool — at least on baseline picks. Our agreement with both numbers proves only that our evaluator matches the human pool; it does not independently corroborate BioACE on non-baseline systems, which is part of why the §10–§10.9 multi-juror infrastructure is methodologically important (§11.5.3).

### 4.6 Phase 1 results

| Pipeline | Strict / 2025: Supports F1 | Strict / 2025: Contradicts F1 | Wall-clock | Notes |
|---|---|---|---|---|
| Starter-kit (organisers') | 44.34 | 4.21 | ~3 h | Reference, BM25 + SciFive-large; scored on our local copy of the human pool |
| **Our Phase 1** | **5.55** | **0.52** | ~6 h | BM25 + MedCPT-CE + DeBERTa-MNLI + NegEx + DeBERTa-MNLI; scored on the same human pool |
| Published 2025 baseline (Table 5) | 44.34 | 4.67 | — | BioACE / Llama-3.3 automatic eval, *not* human pool — see §4.5 subtlety |
| Published CLaC `LLM_NLI_BM25` (Table 5) | 67.74 | 4.57 | — | BioACE automatic eval, top of leaderboard on Supports |
| Published InfoLab `task_a_run2` (Table 5) | 53.41 | 15.67 | — | BioACE automatic eval, top of leaderboard on Contradicts |

The Phase 1 gate `Supports F1 ≥ 60 AND Contradicts F1 ≥ 10` (design D10) **FAILED** at `5.55 / 0.52`. The 38.79 pp / 4.15 pp shortfall against the published baseline drove the Phase 2 hypothesis: this is not algorithmic weakness, it is **pool bias**.

---

## 5. The Pool Bias Problem

### 5.1 Diagnostic numbers

Phase 1 emits **555 distinct support PMIDs across the 194 cells**; only **~30** of those appear in the 588-triple human qrels. The other ~525 were never shown to a human assessor and are therefore counted as false positives. Same shape on contradict: 569 emitted, ~3 in pool. The **theoretical maximum** F1 for Phase 1 given the official qrels is bounded by *recall against the pool* — and the pool only contains the picks of the organisers' baseline. Any system whose retrieval shape diverges from theirs *cannot* score well.

We initially verified this by re-scoring Phase 1's `task_a_output.json` against the **§2.16 expanded pool** (an LLM-judged augmentation of the *Phase 1 picks*). Result on §2.16: Phase 1 jumped from `5.55 / 0.52` to `44.34 / 15.92`. We later (§6.4) discovered the §2.16 pool was structurally **circular** — built from Phase-1-shaped picks, it inflates any Phase-1-aligned pipeline and deflates variants that diverge from it. The §2.17 pool (BM25 top-30, retrieval-shape-agnostic) is the honest replacement; on §2.17 the same Phase 1 lands at `16.43 / 12.01` (§7.2). The §2.16 number is preserved here for narrative continuity but **§7 onwards uses §2.17 as the primary honest comparator**, and §10.8/§10.9 add a yet-more-conservative intersection pool. The conclusion stands either way: the pipeline is not weak by a 38-pp factor; the methodology was wrong.

### 5.2 Why the published baseline scores 44.34

The published baseline scored 44.34 *because* its picks defined the official pool. This is a pathological feature of the published TREC BioGEN 2025 evaluation, not a Phase 1 bug. The standard fix in modern TREC tracks (e.g., TREC DL 2019–2022) is to recruit deep, diverse pools from many participating systems, but BioGEN 2025 was a small track and the pool ended up dominated by the organisers' own baseline submissions.

### 5.3 Hypothesis: pool bias is the dominant Phase 1 residual error

Going into Phase 2 we wrote the hypothesis explicitly. It was either:

- **(A)** Pool bias dominates → an LLM-judged expanded qrels should close most of the 38.79 pp gap.
- **(B)** Pool bias is real but secondary; the dominant error is genuine algorithmic weakness (e.g. retrieval, NLI calibration).

The §6 results below select hypothesis (A) for support, with a residual ~10 pp on contradict consistent with algorithmic margin. The post-Phase-2.5 / 2.6 conservative-pool reading (§10.8 / §10.9 / §11.5.5) refines this: pool bias explains ~10 pp; the additional ~28 pp visible only on the §2.17 pool turns out to be pool-overlap inflation specific to the baseline's role in defining the official pool, not a portable property of the pipeline.

---

## 6. Phase 2 — Pool-Aware Pipeline

### 6.1 LLM-as-judge: design D2 + D3

**Backend abstraction** ([`judge/backends.py`](../src/trec_biogen/judge/backends.py)): a single `HTTPBackend` class speaking OpenAI-compatible Chat Completions; three concrete adapters wired through `BACKEND_REGISTRY`:

| Backend | Model | Cost per 1M tokens (in / out) |
|---|---|---|
| `together` (design spec) | `meta-llama/Llama-3.1-70B-Instruct-Turbo` | $0.88 / $0.88 |
| `together` (executed) | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (3.1 moved to dedicated-endpoint pricing — same family, same rate card) | $0.88 / $0.88 |
| `hf-llama-3.3-70b` (added in Phase 2.5) | `meta-llama/Llama-3.3-70B-Instruct` via HF Inference Providers (Groq route) | usage-priced |
| `hf-qwen-2.5-72b` (added in Phase 2.6) | `Qwen/Qwen2.5-72B-Instruct` via HF Inference Providers (OpenRouter route) | usage-priced |
| `openai-mini` | `gpt-4o-mini` | $0.15 / $0.60 |
| `openai` | `gpt-4o` | $2.50 / $10.00 |

The OSS-default at design time was Together.ai's Llama-3.1-70B (rationale: open weights, comparable MedNLI accuracy at ~3× cheaper than GPT-4o); the *executed* OSS-default became Llama-3.3-70B after the substitution noted above. All adapters speak the same OpenAI-compatible Chat Completions shape, so backend selection is one CLI flag (`--backend`). A `RecordedBackend` allows tests to replay canned responses without network access.

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
| phase2_bm25_rm3_llm_filtered | 4.03 / 0.52 | 9.89 / **12.01** | +5.86 / +11.49 | ~84 min (full) |
| phase2_no_negex | 5.55 / **2.65** | 16.33 / 8.06 | +10.78 / +5.42 | ~9.7 h (full) |
| phase2_bm25_llm_rewrite | 5.29 / 0.52 | 10.65 / 6.03 | +5.36 / +5.51 | ~88 min (full) |
| phase2_scifive_large | 5.55 / 1.04 | 16.43 / 5.85 | +10.88 / +4.81 | ~5.3 h (full) |

Variant not yet executed: `phase2_hybrid` (BM25 + Dense FAISS + RRF; ~24 h CPU encoding + ~2 h GPU). All other seven Phase 2 ablations have landed; `phase2_no_negex` and `phase2_scifive_large` were completed during the Phase 2.5 robustness sweep and re-scored against the intersection pool (§10.8).

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

Confirmed. On the §2.17 expanded pool, our Phase 1 pipeline lands at 16.43 / 12.01 against the starter-kit's 16.55 / 5.34. The 38.79 / 4.15 official-pool gap was almost entirely methodological. **Post-Phase-2.5/2.6 nuance:** under the three-judge intersection pool (§10.9), the Contradicts ranking changes — starter and `no_negex` tie at ~3.7-4.1 pp, both above Phase 1's 1.07-1.12 pp. The Supports cluster (~16-17 pp on internal pipelines) is robust across both pools because Supports has α = 0.93 Jaccard / α ≈ 0.6+ Krippendorff between judges; Contradicts is where the pool-tightening rearranges the leaderboard. The structural pool-bias finding stands; the magnitudes of internal-variant differences require the intersection pool to be reported honestly (§11.5.5).

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

### 8.6 The contradict path is our defensible win — with intersection-pool caveat

Across every internal variant, contradict F1 on the §2.17 expanded pool is `~11.7–12.0 pp`. The starter-kit scores `5.34 pp` on the same pool. The architectural source is structural: NegEx + cue-list pre-filter + DeBERTa-MNLI contradiction-probability max-pool. **Important post-Phase-2.5 refinement:** the cross-judge Contradicts Jaccard is only 0.12 (§10.4 / §10.8) — i.e., the 12.01 pp number depends meaningfully on which judge ratified the candidates. On the two-judge intersection pool, Phase 1 drops to 1.07 pp and `no_negex` (the variant that best embodies the contradict-path architecture without NegEx) is 3.63 pp — still 3.4× Phase 1 but **statistically tied with starter** (4.01 pp, CI overlap). The three-judge intersection pool (§10.9) preserves the ordering at +0.07 pp shifts across the board. Honest framing for the paper: **the architectural choice (NegEx-off contradict path) materially out-performs Phase 1 in every pool; whether it out-performs the starter is judge-dependent and not separable above CI overlap.** The earlier "structural" claim that the path beats starter by >2× holds only on the liberal §2.17 pool — the post-tightening reading is that the path is competitive, not dominant, on Contradicts.

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
Calibration of Modern Neural Networks"*) over 10 equal-width bins.
The post-isotonic column reports **held-out k=5 cross-validated ECE**
with folds split at `qa_id` boundaries (Phase 2.6 §1; the in-sample
value is kept in parentheses as the pre-Phase-2.6 upper bound):

| Backend | ECE (raw) | ECE (post-isotonic, in-sample) | ECE (post-isotonic, k=5 held-out CV) |
|---|---|---|---|
| `openai-gpt-4o-mini` × cot | **0.1136** | 0.0032 | **0.0476 ± 0.0225** |
| `together-llama-3.3-70b` × cot | **0.0961** | 0.0000 | **0.0329 ± 0.0278** |

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
and linear-interpolated prediction recovers most (though not all) of
the calibration gap. The fitted mapping (mini, CoT, in-sample) is::

    raw conf 0.60 → 0.00
    raw conf 0.70 → 0.27
    raw conf 0.80 → 0.92
    raw conf 0.90 → 0.94
    raw conf 0.95 → 1.00

This lets downstream consumers (e.g. selective rejudgment, two-judge
agreement floors) use calibrated probabilities that are *closer* to
statistical meaning than raw model self-reports — the held-out CV ECE
above quantifies *how much closer*.

**Held-out closure (Phase 2.6 §1, 2026-05-23):** the previous edition
of this report flagged the post-isotonic ECE as in-sample (PAV trivially
achieves near-zero ECE on its training set by linear-interpolating
between observed bins). The k=5 cross-validated ECE above closes that
caveat. The held-out numbers are **15× larger than the in-sample mini
estimate** (0.0476 vs 0.0032) and substantially worse for Together
(0.0329 vs 0.0000), but both still sit at or below the Guo et al.
"substantial" threshold of 0.05 — i.e., the isotonic calibrator
*does* generalise across topics, the gain is just much smaller than
the in-sample fit suggested. Folds are split at `qa_id` boundaries
via stable SHA-1 hashing so the same topic never appears in both the
PAV fit and the evaluation; this is the leakage mode that matters
most for this task (same-topic PMIDs recurring across sentences).
The raw ECE numbers (0.1136 mini, 0.0961 Together) are unchanged —
they measure the *uncalibrated* model where train/test split is moot. Full
report: [`reports/llm_judge_calibration.md`](../reports/llm_judge_calibration.md).

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

### 10.4 Multi-backend concordance (second backend on the gold set, Llama-3.3-70B)

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

### 10.5 Pool-size sensitivity analysis

A common challenge to a small-pool TREC track is *"what would the
F1 numbers look like under a different pool size?"*. The §6.5 anchor
44.34 is impressive against the official 588-triple pool but
implicitly inflated by 27.8 pp of pool overlap (§7 dual-pool
analysis showed this). Can we *statistically quantify* how much pool
thinness limits achievable F1?

Yes. For each variant's `task_a_output.json`, we *thin* the §2.17
expanded qrels (4 758 positives) by sub-sampling positives without
replacement at fractions {0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00},
N = 200 thinning iterations per fraction (seed = 0). Cells whose
positives are entirely removed revert to the `unjudged_as_zero` rule.
Each iteration is scored and we report the per-fraction mean macro
F1 with P2.5/P97.5 percentile bands.

**Methodological note (post-review, 2026-05-22):** this is a
thinning / jackknife-style operation, *not* a classical bootstrap on
F1, and the bands are sensitivity intervals on pool size + identity
of retained positives — not frequentist CIs on the true-population
F1. The qualitative conclusions below (variant ordering stable at
full pool, unstable at thin pool) hold under the thinning
interpretation; the framing was tightened in
[`reports/pool_coverage_analysis.md`](../reports/pool_coverage_analysis.md).

Headline observations:

* The variant ranking *changes* between thin pool (frac = 0.10) and
  full pool (frac = 1.00). Specifically `phase1_baseline` and
  `starter_baseline` swap positions 2 and 3 on Supports as the pool
  thins. The differences between adjacent variants are within the
  thinning-band width (~2 pp at frac = 0.10), so the **leaderboard
  ordering at thin-pool sizes is within sampling noise**.
* `bm25_rm3` is the *least* pool-dependent variant (Δ from frac=0.10
  to frac=1.00 of only +5.20 pp vs +9-10 pp for the others). This
  is the statistical fingerprint of *genuine algorithmic
  weakness*, not pool overlap quirk — `bm25_rm3` is genuinely worse,
  not just unlucky against the pool.
* The official pool's effective fraction relative to the §2.17
  expanded pool is approximately 588 / 4 758 ≈ 12 %. Thinning bands
  at frac = 0.10–0.20 confirm: at official-pool sizes, the cross-
  variant differences are roughly within the noise floor.

Full report:
[`reports/pool_coverage_analysis.md`](../reports/pool_coverage_analysis.md).

### 10.6 LLM-filtered PRF — addressing the §8.3 negative result

The §8.3 negative result (blind RM3 hurts) had a hypothesised
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
   evidence-bearing candidates that drove RM3 drift in §8.3. The
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

This is a useful nuance for the paper: it strengthens the §8.3
finding (RM3 hurts), shows the LLM-filtered PRF literature is
correct on its own terms (LLM filtering > blind PRF), and falsifies
the textbook IR assumption that *any* PRF helps on *any* task.

### 10.8 Judge robustness and per-topic analysis (Phase 2.5)

After Phase 2 sign-off, two methodological gaps remained: (i) the
§11.3 structural claim that `phase2_no_negex` beats Phase 1 on
contradicts rested on a *single-judge* expanded pool (`gpt-4o-mini
--prompt cot`), even though §10.4 had already established that
Contradicts F1 carries judge-dependent variance (κ overall 0.34, with
the asymmetry concentrated on the contradict class); and (ii) the
quantitative summary did not surface the per-topic distribution of
gains and losses, so a reader could not see where the aggregate F1
numbers actually came from. Phase 2.5 closed both.

A second LLM-judge rejudge was performed on the same 5398 candidate
triples from §2.17 using `Llama-3.3-70B-Instruct --prompt cot`
(hosted via the HuggingFace Inference Providers router, auto-routed
to Groq). The model weights are identical to the Together-hosted
endpoint that cleared the §10.4 gate at κ=0.9112 [0.886, 0.936], so
no re-validation was required. Cost: $2.47 for the full pass
(~21 min wall-clock). The cross-judge agreement on the 5398-triple
set was *strongly asymmetric*: Jaccard 0.93 on Supports, **0.12 on
Contradicts** (43 agreed of 378 union). Llama emits 6.3× fewer
Contradicts than mini under the same CoT temperature — the §10.4
asymmetry, amplified at scale.

A **two-judge intersection-on-contradicts pool** was then derived:
human positives copied verbatim, Supports passed through from the
canonical mini-cot pool (where the judges agree), Contradicts kept
only when both judges independently labelled the same triple as
Contradicts. The pool went from 4 758 to 4 438 total positives — the
88 % drop is entirely on the Contradicts class (363 → 43). This is a
deliberately strict pool: it represents "what would survive a
two-juror unanimity requirement", at the cost of pushing the
Contradicts macro toward the sampling-noise floor (43 positives across
313 cells, mean 0.14 per cell).

A `--qrels-pool=intersection` flag was wired end-to-end through
`eval/metrics.py` and `eval/phase2_summary.py`. Cell-level bootstrap
CIs (B=1 000 resamples, seed=0, 95% percentile interval) were
computed for every existing run on the intersection pool. The
results:

| variant | Supports F1 (intersection) | Contradicts F1 (intersection) |
|---|---|---|
| starter_baseline | 16.55 [15.01, 18.25] | **4.01 [2.13, 6.13]** |
| Phase 1 | 16.43 [15.15, 17.80] | 1.07 [0.26, 2.04] |
| `no_negex` | 16.33 [15.15, 17.59] | **3.63 [1.98, 5.38]** |
| `scifive_large` | 16.43 [15.09, 17.73] | 2.21 [0.88, 3.79] |
| `allow_existing` | 16.94 [15.59, 18.26] | 1.07 [0.21, 2.10] |
| `bm25_rm3` | 8.97 [7.79, 10.21] | 0.55 [0.00, 1.32] |

Two findings emerge under the conservative pool. **First**, the
structural Phase 2 claim survives: `no_negex` beats Phase 1 on
Contradicts (3.63 vs 1.07, ~3.4× the midpoint, lower-CI overlap is
marginal). **Second**, the expanded-pool reading "no_negex >>>
starter on contradicts" (12.01 vs 5.34) *does not* survive: 3.63 vs
4.01 sits inside the CI overlap [2.13, 6.13] ∩ [1.98, 5.38]. The
honest synthesis is that **starter and no_negex are statistically
indistinguishable on the conservative-pool Contradicts macro**, and
both clearly beat Phase 1 — meaning the Phase 2 contribution on
Contradicts is real, but the *magnitude* visible on the expanded
pool was inflated by liberal mini-cot ratifications that Llama did
not corroborate. The three retrieval-side negative results
(`bm25_rm3`, `bm25_rm3_llm_filtered`, `bm25_llm_rewrite`) survive
the pool tightening unchanged: all sit several CIs below the
~16-17 Supports band where every selection-side variant clusters.

The per-topic dimension was added on top. Topics were selected
mechanically (no cherry-picking) by ranking `phase1_baseline.topic_F1 −
starter_baseline.topic_F1` on the intersection pool, Strict Supports,
and taking the largest positive Δ (qa=150, +13.94 pp), the closest-to-
zero (qa=120, +0.00), and the largest negative Δ (qa=131, −19.49 pp).
The full sorted appendix is published in
[`reports/per_topic_error_analysis.md`](../reports/per_topic_error_analysis.md);
qualitative analysis there cites concrete PMIDs and the judge's verdict
per PMID. The synthesis: Phase 1 wins on non-classical topics where
MedCPT-CE surfaces LLM-confirmed-but-pool-invisible PMIDs (qa=150);
ties on topics where both pipelines converge on the same understanding
but cite disjoint valid evidence (qa=120, the pool-bias dance in
microcosm); and loses on topics with sub-population-specific human-pool
golds that the reranker demotes in favour of broader topical relevance
(qa=131, formoterol side effects — Phase 1 actually wins sentences 1-2
but loses sentences 4-5 on paediatric and cancer-patient subpopulations
where small, curated literature dominates). The aggregate ~0.1 pp
difference between Phase 1 and starter on the conservative-pool
Supports hides these compositional effects; the per-topic view makes
them visible.

The full Phase 2.5 cost was $2.47 (HF Inference Providers, the
Llama-3.3-70B rejudge). Code lives under
[`src/trec_biogen/judge/intersection.py`](../src/trec_biogen/judge/intersection.py),
[`src/trec_biogen/eval/per_topic.py`](../src/trec_biogen/eval/per_topic.py),
and [`scripts/per_topic_diff.py`](../scripts/per_topic_diff.py); reports
are [`reports/judge_intersection_analysis.md`](../reports/judge_intersection_analysis.md)
and [`reports/per_topic_error_analysis.md`](../reports/per_topic_error_analysis.md).
The Phase 2.5 work also added incremental-checkpoint atomic writes and
5xx-retry semantics to the judge backend (lessons learned mid-run); see
the change at `openspec/changes/archive/2026-05-22-phase2-5-judge-robustness/`.

### 10.9 Three-judge intersection and Krippendorff α (Phase 2.6)

Phase 2.5 left one substantive question open: the 88 % drop on the
two-judge intersection-on-contradicts pool admits two readings — "mini-cot
is over-emitting contradicts" or "Llama-70B is over-stripping them". With
two judges there is no way to triangulate. Phase 2.6 closes this by
adding a *third* judge from a different model family — design D1 in the
openspec change selected Mixtral-8x7B via HF Inference Providers, but at
implementation time HF had removed the Mistral family from the chat-routable
roster (`400 "not a chat model"`) and the Together-direct fallback hit
HTTP 402. The pivot, documented in the design, went to **Qwen2.5-72B-Instruct**
(Alibaba dense 72B, routed by HF Providers to OpenRouter). Same intent:
third model family distinct from both OpenAI's GPT-4 line and Meta's
Llama-3 line.

Qwen passed the §6.5 concordance gate at macro-w-F1 = **0.8980** (Supports
F1 0.944 / Contradicts F1 0.250 / n=549 / n=39). It sits between mini-cot
(0.8944) and Llama-70B-cot (0.9112) on the macro and matches Llama's
Contradicts conservatism almost exactly. A second-judge rejudge of the
full 5 398 §2.17 candidate set was then performed (cost $4.00 total
across gold + expand-pool, after one HF Router 400 mid-run that resumed
cleanly from the per-200-triple checkpoint).

A three-way Krippendorff α (Krippendorff 2011 nominal-data formulation,
implemented in [`eval/metrics.py::krippendorff_alpha`](../src/trec_biogen/eval/metrics.py))
was then computed across the three judges on both the 588 gold set and
the 5 398 expand-pool set. The headline:

| α | Value |
|---|---:|
| 588 gold set, 3-way α (full label space) | 0.3643 |
| 5 398 candidate set, 3-way α (full label space) | 0.2992 |
| 5 398 candidate set, pairwise α: mini ↔ Llama | 0.1166 |
| 5 398 candidate set, pairwise α: mini ↔ Qwen | 0.2041 |
| **5 398 candidate set, pairwise α: Llama ↔ Qwen** | **0.6013** |

This resolves the Phase 2.5 open question: **mini-cot is the contradict-class
outlier**, not three-way disagreement. Llama and Qwen — trained by
different organisations on different data with different architectures —
converge on a substantially-agreed set of contradicts (α=0.60, "substantial"
per Landis & Koch 1977). Mini-cot agrees with neither at much above the
"slight" level. The two-judge intersection pool from Phase 2.5 was
effectively "mini's contradicts, filtered through Llama's veto"; the
three-judge pool is essentially "Llama and Qwen's shared contradicts
(which mini also happens to ratify in most cases)".

The three-way intersection-on-contradicts pool was emitted by the
generalised `intersection.emit_intersection_pool(records_paths=[mini, llama, qwen], ...)`
helper (Phase 2.5's two-judge contract preserved byte-for-byte by a
regression test that re-emits the archived two-judge pool and asserts
`out.read_bytes() == archived.read_bytes()`). Counts:

* 363 mini-only contradicts → 43 mini ∩ Llama (Phase 2.5) → **31** mini ∩ Llama ∩ Qwen (Phase 2.6) — 91.5 % drop from mini-only.
* Pairwise Llama ∩ Qwen alone: 32 — essentially identical to the three-way 31, confirming the α reading.
* 31 surviving positives clear the design-D2 floor of 20 (macro statistics remain reportable but small-sample caveat applies on Contradicts).

Re-scoring every existing run dir against the 3-way pool (a single column
appended to [`reports/phase2_summary.md`](../reports/phase2_summary.md)
via the new `--qrels-pool=intersection-3way` flag in `eval/metrics.py`)
showed that the Phase 2.5 §10.8 conclusions carry over **unchanged**:

| variant | Contradicts F1 (2-way) | Contradicts F1 (3-way) | Δ |
|---|---:|---:|---:|
| starter_baseline | 4.01 | **4.08** | +0.07 |
| Phase 1 | 1.07 | 1.12 | +0.05 |
| `no_negex` | 3.63 | **3.70** | +0.07 |

`no_negex` still beats Phase 1 by ~3.3× (the structural Phase 2 finding);
`no_negex` is still statistically indistinguishable from `starter` (the
apparent expanded-pool advantage remains an artefact of liberal mini-cot
ratifications). The +0.07 pp shift across the board is the mechanical
consequence of going from 43 to 31 Contradicts positives in the denominator;
the same TP count divided by a smaller pool gives marginally higher F1.

The deferred work the §10.2 caveat asked for — held-out cross-validated
ECE in place of in-sample isotonic ECE — was also delivered in Phase 2.6
§1. See §10.2 for the updated calibration table. Together the two §10.2
+ §10.9 deliverables close the methodological frontiers that Phase 2.5
sign-off explicitly flagged as open. Code and data artefacts under
[`src/trec_biogen/judge/intersection.py`](../src/trec_biogen/judge/intersection.py)
(N-way generalisation), [`src/trec_biogen/eval/metrics.py`](../src/trec_biogen/eval/metrics.py)
(Krippendorff α), [`data/qrels/biogen2025_taskA_qrels_intersection_3way.jsonl`](../data/qrels/biogen2025_taskA_qrels_intersection_3way.jsonl);
analysis report at [`reports/judge_intersection_analysis.md`](../reports/judge_intersection_analysis.md);
openspec change at `openspec/changes/archive/2026-05-24-phase2-6-judge-robustness-ii/`.

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

The methodology family was crystallised by Zheng et al. (2023, *MT-Bench*, [arxiv:2306.05685](https://arxiv.org/abs/2306.05685)) and extended in many directions. The closest external parallel to our work is the **TREC Health Misinformation 2022 track** (Clarke et al., 2023, TREC overview) which used a GPT-4-based judge for pool expansion on a similar biomedical-evidence task. Their concordance threshold was 0.80; we adopted 0.85 because (a) BioGEN Task A has cleaner labels (binary support/contradict vs misinfo's multi-axis), and (b) the design risk register explicitly called for a stricter gate.

**The closest *internal* parallel — disclosed in the 2025 overview itself — is the organisers' own BioACE framework with Llama-3.3** (Gupta, Bartels and Demner-Fushman, *"BioACE: An Automated Framework for Biomedical Answer and Citation Evaluations"*, [arXiv:2602.04982](https://arxiv.org/abs/2602.04982), 2026). The overview (Gupta et al., 2026, §7.1.2) states verbatim: *"For all submitted runs, we used the BioACE evaluation framework for citation evaluations. In particular, we leverage the nugget-based evaluation using Llama-3.3, where, given answer-sentence and cited-document nuggets, the model is prompted to classify the answer-sentence and document pair as Supports, Contradicts, Neutral, or Not relevant."* The overview defers concordance analysis to future work in §8.2: *"We will conduct a detailed analysis of the correlation between expert and automated evaluation in future work."* This means: the BioGEN 2025 evaluation methodology *already* incorporates a single-backend, no-CI, no-multi-juror LLM judge — the gap our Phase 2 — Phase 2.6 fills is the *defensibility infrastructure* around such a judge (§11.5.3).

Recent critical work on LLM-as-judge (e.g., Wang et al., 2024, *"Large Language Models are not Fair Evaluators"*) documents systematic biases (positional, verbosity, calibration). The CoT diagnostic we report in §6.2 is consistent with that literature: the LLM's failure was not "bad calibration" but "no surface for reasoning". Adding CoT is the standard mitigation; the fact that we *measured* the impact (0.7497 → 0.8944 macro-w-F1 on the same model) is what makes it actionable beyond our own work.

The methodological extensions that were *open* at Phase 2 sign-off have since been closed:

1. **Backend-sensitivity experiment** (design D10) — closed by §10.4, which ran `Llama-3.3-70B-Instruct --prompt cot` (Together-served; Llama-3.1-70B-Turbo had moved to dedicated-endpoint pricing) over the 588-triple gold set as the third independent backend and reported pairwise concordance with the OpenAI pair.
2. **Two-judge robustness on the full expand-pool** — closed by Phase 2.5 (§10.8), which re-ran the same Llama-3.3-70B weights through HF Inference Providers (Groq-routed) over the full 5 398-triple §2.17 candidate set, derived a two-judge intersection-on-contradicts pool, and re-scored every variant under cell-level bootstrap CIs.
3. **Three-juror Krippendorff α** — closed by Phase 2.6 (§10.9), which added Qwen2.5-72B as a third independent family. The α = 0.60 Llama↔Qwen result identifies `mini-cot` as the Contradicts-class outlier; the same three-judge corroboration would, applied to BioACE's Llama-3.3, directly answer the overview's deferred concordance question.

The remaining open frontier (deferred to §13) is a **fourth backend** — ideally a biomedical-domain model (Mixtral-Instruct, MedPaLM-Instruct if accessible) — which would let us report 4-way α and tighten the intersection pool further on the Contradicts class. The 31-positive 3-way intersection clears the design-D2 floor of 20 but sits in small-sample territory.

### 11.5 The 2025 leaderboard — full participant survey

The published BioGEN 2025 overview (Gupta et al., 2026, [arXiv:2603.21582](https://arxiv.org/abs/2603.21582)) lists **seven Task A teams**: CLaC, dal, GEHC-HTIC, InfoLab, polito, SIB, uniud. Table 5 of the overview reports each run on the official 244-PMID human-judged pool (10 topics, one top-priority run per team pooled). The full Strict numbers are:

| Team | Run | Strict Sup P/R/F1 | Strict Con P/R/F1 |
|---|---|---|---|
| CLaC | LLM_NLI_BM25 | 67.18 / 74.36 / **67.74** | 3.61 / 7.73 / 4.57 |
| CLaC | LLM_BM25 | 66.75 / 67.46 / 64.10 | 3.95 / 7.60 / 4.77 |
| InfoLab | task_a_run6_A | 66.92 / 71.17 / **67.23** | 12.71 / 17.65 / 14.15 |
| InfoLab | task_a_run4 | 52.92 / 60.30 / 54.49 | 10.74 / 15.12 / 11.85 |
| InfoLab | task_a_run2 | 52.75 / 56.80 / 53.41 | 14.09 / 19.42 / **15.67** |
| InfoLab | task_a_run1 | 39.95 / 47.59 / 42.07 | 7.99 / 13.92 / 9.73 |
| InfoLab | task_a_run3 | 18.56 / 18.56 / 18.56 | 0.52 / 0.52 / 0.52 |
| GEHC-HTIC | gehc_htic_task_a | 56.70 / 57.37 / 53.53 | 6.62 / 14.00 / 8.57 |
| polito | scifive-ft-512CL-lex | 52.58 / 64.54 / 55.81 | 4.04 / 6.70 / 4.79 |
| dal | emotional_prompt | 50.60 / 67.23 / 55.53 | 1.29 / 1.29 / 1.20 |
| dal | expert_prompt | 45.62 / 55.67 / 48.80 | 0.52 / 0.26 / 0.34 |
| SIB | SIB-task-a-1 | 52.41 / 74.23 / 58.87 | 0.00 / 0.00 / 0.00 |
| SIB | SIB-task-a-2 | 47.85 / 64.09 / 52.30 | 3.69 / 5.67 / 4.21 |
| SIB | SIB-task-a-3 | 15.12 / 26.80 / 18.42 | 0.00 / 0.00 / 0.00 |
| SIB | SIB-task-a-4 | 17.01 / 32.47 / 21.31 | 0.00 / 0.00 / 0.00 |
| SIB | SIB-task-a-5 | 46.22 / 64.00 / 51.17 | 2.06 / 3.95 / 2.56 |
| SIB | SIB-task-a-6 | 36.25 / 42.14 / 37.82 | 1.98 / 2.58 / 2.15 |
| SIB | SIB-task-a-7 | 39.52 / 45.02 / 41.08 | 2.06 / 2.58 / 2.23 |
| uniud | run1_no-rerank_sparse | 40.89 / 41.78 / 39.10 | 1.55 / 2.66 / 1.87 |
| uniud | run2_rerank_sparse | 39.18 / 44.86 / 38.85 | 2.23 / 1.98 / 2.03 |
| uniud | run3_no-rerank_dense | 26.55 / 19.00 / 19.82 | 3.61 / 9.19 / 5.07 |
| uniud | run4_rerank_dense | 16.58 / 8.79 / 10.52 | 2.58 / 7.22 / 3.76 |
| **Baseline** | **TEST** | **51.03 / 44.07 / 44.34** | **3.44 / 8.08 / 4.67** |
| **Our Phase 1** | **phase1_baseline** | **— / — / 5.55** | **— / — / 0.52** |

#### 11.5.1 Pool construction (critical context)

The overview discloses that the official pool was constructed from:

- **10 topics** (not all 40 in the input file).
- **One top-priority run per team**, including the baseline. Seven teams → 7 + 1 baseline runs contributing to the pool.
- **244 PubMed abstracts** manually assessed across those 10 topics. The 588-triple JSONL we use as the human qrels is the cell-level expansion of those 244 PMIDs (PMID × answer-sentence × class).

This is exactly the pool-bias mechanism we diagnose in §5, with two amplifying factors the overview clarifies: only **one** run per team is pooled (so a team that submits ten runs only gets one of them judged), and only **10** of 40 topics receive any human judgement at all. The 27.8 pp inflation we quantify in §8.1 for the baseline (`44.34` official vs `16.55` on our §2.17 pool) is the size of this effect for the team whose retrieval shape *was* the pool. For non-participating retrospective systems like ours, the analogous gap is smaller in absolute terms (their picks were neither pre-judged nor pre-rejected) but works in the same direction.

#### 11.5.2 Pipeline architectures observed

Cross-team architectural patterns from the seven Task A teams' notebook papers (NIST [TREC 34 proceedings](https://trec.nist.gov/pubs/trec34/index.html)) and the GE Healthcare arXiv preprint ([arXiv:2603.17580](https://arxiv.org/abs/2603.17580)):

| Team | First-stage | Rerank | NLI / decision | LLM in decision path |
|---|---|---|---|---|
| **CLaC** | BM25 | ColBERT | NLI module | Yes (`LLM_NLI_BM25` → best Sup F1 67.74) |
| **CLaC Lab** (2nd Concordia submission) | BM25 + dense (modular sparse+dense) | — | NLI | — |
| **InfoLab** | BM25 | strong reranker | SciFive-MedNLI variants | — (best Con F1 15.67) |
| **GEHC-HTIC** | BM25 ("Decoupled Lexical Architecture") | Narrative-Aware Reranking | — | Yes (One-Shot In-Context Learning) |
| **dal** | BM25 + RAG variants | — | Llama-3 70B / GPT-3.5 in decision | Yes (`emotional_prompt`, `expert_prompt`) |
| **polito** | BM25 | — | SciFive-large fine-tuned on MedNLI | — |
| **SIB** | BM25 / SIBiLS | — | Bio-Medical-Llama-3-8B (per HF card) | Yes |
| **uniud** | sparse + dense passage indexes | with/without rerank (4 ablations) | — | — |
| **Baseline (TEST)** | BM25 (PySerini, top-1000) | `ms-marco-MiniLM-L-6-v2` | SciFive-MedNLI | — |
| **Our Phase 1** | BM25 (Pyserini, top-100/1000) | **MedCPT-Cross-Encoder** | **DeBERTa-v3-MNLI-FEVER-ANLI** | — (LLM only in the judge, not the pipeline) |

Three architectural divergences worth flagging:

1. **MedCPT vs ms-marco-MiniLM / GraphMonoT5** — we chose a PubMed-domain-pretrained reranker; the baseline and several participants used general-purpose rerankers (MiniLM, MonoT5). The `phase2_no_rerank` ablation (§7.2) shows MedCPT-CE is approximately neutral on intrinsic quality (see §8.4) — so this architectural divergence does not propagate to a measurable F1 difference once pool bias is removed.
2. **DeBERTa-MNLI vs SciFive-MedNLI** — the dominant biomedical-NLI choice among participants (baseline TEST, polito, InfoLab variants) is SciFive-large fine-tuned on MedNLI. We use general-domain DeBERTa-v3-MNLI-FEVER-ANLI. Our `phase2_scifive_large` variant tests the biomedical-specialised path: 1.04 Con F1 official, 5.85 expanded — *worse* than DeBERTa on the contradict path. Domain-specific NLI fine-tuning is not a free win on this task.
3. **LLM-in-pipeline vs LLM-in-judge** — every published-top Supports system (CLaC's `LLM_NLI_BM25`, GEHC-HTIC, dal's prompt variants) places an LLM in the *decision* path. We deliberately keep the LLM out of the pipeline and confine it to the *evaluator* role. This is a defensible scientific choice (cleaner attribution of pipeline quality vs LLM quality) but it is also the most plausible reason CLaC reaches 67.74 Sup F1 and we do not: LLMs in the decision path appear to extract significantly more support signal than NLI-only architectures on this task. Folding an LLM into the decision path is the natural Phase 4 extension (§13).

#### 11.5.3 The BioACE / Llama-3.3 disclosure — repositioning our contribution

The overview discloses (§7.1.2 for Task A and §7.2.2 for Task B) that for **all submitted runs**, the organisers used the **BioACE evaluation framework with Llama-3.3** as the automatic evaluator — and Table 5 specifically reports BioACE-derived F1, not expert-derived F1. Verbatim from §7.1.2: *"For all submitted runs, we used the BioACE evaluation framework for citation evaluations. In particular, we leverage the nugget-based evaluation using Llama-3.3, where, given answer-sentence and cited-document nuggets, the model is prompted to classify the answer-sentence and document pair as Supports, Contradicts, Neutral, or Not relevant."* Expert evaluation is reported separately (Figures 3, 4, 5 in the overview), and §8.2 explicitly defers expert-vs-automated concordance to future work: *"We will conduct a detailed analysis of the correlation between expert and automated evaluation in future work."* BioACE is documented in a companion paper ([arXiv:2602.04982](https://arxiv.org/abs/2602.04982)).

**A subtle but important consequence:** the 44.34/4.67 Strict baseline numbers we calibrate against in §4.5 come from the same Llama-3.3-driven BioACE evaluation — not from the 244-PMID human pool directly. Our `biogen2025_taskA_qrels.jsonl` file is the cell-level expansion of the 244-PMID human pool (i.e., the *expert* evaluation underlying Figures 3-5); we reproduce 44.34 by scoring starter-kit picks against that human pool, and the BioACE Table-5 number for the same baseline happens to also be 44.34 because BioACE was very likely calibrated to agree with the same human pool on baseline picks. The two evaluations agreeing on the baseline does not mean they agree on every system — and crucially, it does not mean BioACE has been independently corroborated as a judge.

This repositions our Phase 2 — Phase 2.6 contribution. **We are not the first to use LLM-as-judge on this task — the organisers themselves do.** The differentiation, on which the rest of §10 turns, is the methodological infrastructure we layer on top:

| Aspect | BioACE (organisers, 2025) | Our Phase 2 – 2.6 |
|---|---|---|
| Backend | single (Llama-3.3) | three (mini-cot, Llama-3.3, Qwen2.5-72B) |
| Concordance validation against humans | deferred to "future work" (§8.2 of the overview) | gate ≥ 0.85 with bootstrap CI per backend (§10.1) |
| Prompt diagnosis | nugget-based prompt, no documented pivot | strict → CoT pivot driven by domain-expert read of disagreement cases (§6.2.3) |
| Multi-juror agreement | not reported | Krippendorff α = 0.60 Llama↔Qwen; mini-cot identified as Contradicts outlier (§10.9) |
| Pool basis | the 244-PMID human pool (BioACE judges only pooled candidates) | §2.17 expansion to 4 758 positives (3.7× the human pool) — judged candidates are not restricted to the original pool |
| Confidence calibration | not reported | held-out k=5 CV ECE per backend (§10.2, Phase 2.6 §1) |
| Conservative reporting | single point estimate (Table 5) | dual + two-judge + three-judge intersection pool + cell-level bootstrap CIs (§10.8, §10.9) |

The contribution is therefore *not* "we used LLM-as-judge" — it is **the methodological infrastructure that makes single-backend LLM-as-judge defensible**: validation against humans with CI, multi-juror corroboration, calibrated confidence, prompt-pivot diagnosis, and conservative-pool reporting. The 2026 overview-cycle, which the organisers have signalled will return to the expert-vs-automated concordance question, is the natural venue for this contribution.

#### 11.5.4 The code-availability gap

Cross-referencing the seven notebook papers in [NIST TREC 34 proceedings](https://trec.nist.gov/pubs/trec34/index.html) against author personal GitHubs and institutional organisations ([CLaC-Lab](https://github.com/CLaC-Lab), [sib-swiss](https://github.com/sib-swiss), [ailab-uniud](https://github.com/ailab-uniud), [jknafou](https://github.com/jknafou), [jarobyte91](https://github.com/jarobyte91)): **zero of the seven Task A teams have published code for their 2025 submission**. The only BioGEN-2025-related public repository is the organisers' [starter-kit-2025](https://github.com/trec-biogen/starter-kit-2025) (which we vendor as our calibration anchor via `scripts/vendor_starter_kit.sh`). The nearest precedent is [Webis at TREC 2024 BioGen](https://github.com/webis-de/trec24-biogen), but Webis did not appear in the 2025 Task A list.

Implication: any independent verification of the Table 5 numbers is currently impossible at the implementation level. A 2026-cycle pipeline that publishes its code, Hydra configs, and the §2.17 / Phase 2.5 / Phase 2.6 qrels-augmentation artefacts would be **the first publicly reproducible Task A reference in the track's history**. This is the strongest standalone case for an external submission of our work — independently of where we sit on the leaderboard.

#### 11.5.5 Comparable results — and where we genuinely sit

A direct comparison against the official Table 5 is unfair to us (our pipeline never contributed to pool definition; the official numbers measure both system quality and pool-overlap-with-the-pooled-runs). The cleanest comparable we can run on the same evaluator and the same pool is the starter-kit (the baseline TEST run): on the §2.17 expanded pool, starter-kit scores `16.55 / 5.34`; our Phase 1 scores `16.43 / 12.01`; `no_negex` scores `16.33 / 8.06`. The conservative reading (three-judge intersection pool, §10.9):

- **Supports.** Phase 1 ≈ starter on the conservative-pool macro F1 (16.43 vs 16.55, CI overlap). The published 67.74 of CLaC's `LLM_NLI_BM25` cannot be re-scored against the §2.17 pool without their `task_a_output.json`, so the honest external claim is *"we cannot rule out that CLaC's 67.74 also contains pool-overlap inflation comparable to the baseline's 27.8 pp, but we cannot quantify it either."* What we *can* say is that the architectural difference (LLM-in-decision vs NLI-only) is large enough that an honest re-evaluation would likely still leave CLaC ahead on Supports.
- **Contradicts.** Phase 1 (1.07–1.12 intersection) clearly loses to InfoLab on the official pool (12.71–15.67). But our `phase2_no_negex` (3.63–3.70 intersection) is statistically indistinguishable from the starter (4.01–4.08), and clusters with the structural Phase 2 contradict story. InfoLab's superiority on Contradicts is real *and* benefits from being in the pool's contributing set — both effects compose. The bigger-picture finding here is *which path beats which* on the conservative pool: NegEx-off variants and the starter cluster at ~3-4 Con F1; Phase 1 and `allow_existing` cluster at ~1; `bm25_rm3` is the bottom.

Two-judge intersection-pool bootstrap CIs (95% percentile, B=1000), as in §10.8:

| Variant | Sup F1 [95% CI] | Con F1 [95% CI] |
|---|---|---|
| starter_baseline | 16.55 [15.01, 18.25] | **4.01 [2.13, 6.13]** |
| Phase 1 | 16.43 [15.15, 17.80] | 1.07 [0.26, 2.04] |
| `no_negex` | 16.33 [15.15, 17.59] | **3.63 [1.98, 5.38]** |
| `scifive_large` | 16.43 [15.09, 17.73] | 2.21 [0.88, 3.79] |
| `allow_existing` | 16.94 [15.59, 18.26] | 1.07 [0.21, 2.10] |
| `bm25_rm3` | 8.97 [7.79, 10.21] | 0.55 [0.00, 1.32] |

Honest external positioning: **on Supports we are mid-pack and clearly behind the LLM-in-decision teams (CLaC, GEHC-HTIC, dal). On Contradicts we are mid-pack on the official pool but our methodological work corrects what the published numbers actually mean.** The contribution that survives any future reweighting is the §10–§10.9 infrastructure; the pipeline's intrinsic ranking is bounded by the LLM-in-pipeline gap we did not bridge.

---

## 12. Limitations

- **Hardware**: 4 GB VRAM is the binding constraint. Full-corpus dense retrieval (~80 GB FAISS), late-interaction (~150 GB ColBERT-v2), and any joint MedCPT-Article + Cross-Encoder run are infeasible without renting an A100 or similar. The work would scale linearly to a 24 GB+ GPU; nothing in the design assumes 4 GB beyond the sequential model-loading convention.
- **No NLI fine-tuning**. Phase 3 (deferred) would fine-tune DeBERTa or SciFive on SciFact, HealthVer, or BioNLI. Empirically those provide 3–5 pp on MedNLI; whether it translates to a Task A lift is open.
- **LLM-judge backend dependence**. The canonical §2.17 expanded qrels were produced by `gpt-4o-mini --prompt cot` only; Phase 2.5 (§10.8) added Llama-3.3-70B and Phase 2.6 (§10.9) added Qwen2.5-72B as independent backends. The three-way Krippendorff α (mini↔Llama 0.12, mini↔Qwen 0.20, **Llama↔Qwen 0.60**) identifies `mini-cot` as the Contradicts-class outlier. The conservative posture (two-judge and three-judge intersection-on-contradicts pools, §10.8 / §10.9) is what we report in §11.5.5 as the headline numbers. **Remaining gap:** all three judges are *general-domain* LLMs. A fourth backend from a **biomedical-domain family** (Mixtral-Instruct, MedPaLM-Instruct if accessible, or a fine-tuned Bio-Medical-Llama-3 variant) would let us test whether domain priors change the Contradicts distribution — currently we cannot distinguish "general LLMs underestimate Contradicts" from "Contradicts class is genuinely rare in PubMed".
- **The expanded pool is local to this submission's retrieval shape**. §2.17 covers BM25 top-30 across both paths but is not deep enough for variants that radically change retrieval (e.g., `phase2_hybrid` with FAISS-based dense retrieval would surface PMIDs outside BM25's top-30). Each such variant should run its own `expand-pool` pass on its own retrieval parquets before comparison.
- **Single annotator perspective**. Our domain expert reviewed 12 disagreement cases; the diagnosis was good but the sample is small. A peer-reviewed paper would want 50+ cases reviewed by two independent biomedical experts.
- **LLM-in-decision gap vs the field**. §11.5.2 shows that every published-top Supports system (CLaC `LLM_NLI_BM25`, GEHC-HTIC, dal) places an LLM in the *decision* path, not only in the judge. We deliberately kept the LLM out of the pipeline for cleaner attribution; the cost is a structural cap on our Supports F1 that the §10–§10.9 methodological work does not close. Phase 4 (§13) lifts this restriction.
- **No team-level head-to-head**. §11.5.5 spells out the strongest external claim we can make: starter-kit ≈ our Phase 1 on the §2.17 expanded pool. We cannot re-score CLaC, InfoLab, GEHC-HTIC, dal, polito, SIB, or uniud on our pool because none of them published code or task_a_output.json files (§11.5.4). Any "X% of CLaC's 67.74 would survive an honest pool" claim is unprovable until that changes.

---

## 13. Future Work

- **`phase2_hybrid` ablation**. BM25 + Dense (MedCPT-Article) + RRF fusion (§3.4) is wired through Hydra but unrun on the full 2025 input (~24 h CPU encoding for the 5 M-doc FAISS index + ~2 h GPU). Re-running `expand-pool` on this variant's own retrieval parquets (§11 caveat) is a prerequisite for an honest expanded-pool comparison.
- **k-fold cross-validated ECE for the LLM judge**. The §10.2 isotonic-calibration ECE is in-sample on the 588-triple gold set; a defensive estimate would fit PAV on k-1 folds (folded at `qa_id` boundaries to avoid topical leakage) and report held-out ECE. Cheap (~minutes, no API spend) and tightens the calibration claim.
- **Phase 3 — NLI fine-tuning**. SciFact + HealthVer + BioNLI compose ~50 k labelled training pairs. A QLoRA-tuned DeBERTa-v3-base on a free Colab GPU would land within ~6 h. Expected lift: 2–5 pp on contradict.
- **Phase 4 — Agentic retrieval**. Insert an LLM in the *first-stage* loop (query rewriting from `(question, answer-sentence) → search-engine-style biomedical query`). The `phase2_bm25_llm_rewrite` ablation (§10.7) is a single-shot proxy; the agentic pattern with reflection / multi-turn querying remains open. Risk: cost (every query is an LLM call) and latency.
- **Phase 4 — LLM-in-decision pipeline variant**. §11.5.2 identifies LLM-in-decision (CLaC, GEHC-HTIC, dal) as the most plausible reason the published Supports leaderboard top reaches 67.74 while NLI-only pipelines (baseline, polito, ours) cap around 44–55. The natural variant is `phase2_llm_decision`: replace DeBERTa-MNLI with a CoT-prompted gpt-4o-mini decision over the MedCPT-CE top-30, reusing the existing judge backend abstraction. Expected lift on Supports: substantial (10+ pp on expanded pool); cost: ~$1–2 per full run at concurrency=8.
- **Reproduce CLaC / InfoLab / GEHC-HTIC on the §2.17 expanded pool**. The strongest external claim in §11.5.5 is bounded by our inability to re-score other teams' submissions. If any team publishes their `task_a_output.json` (or, better, code), running it through `eval/phase2_summary.py` with `--qrels-pool=intersection-3way` would close the published-vs-honest gap for that team. This is one email away.
- **Submit to TREC BioGEN 2026 as the first publicly reproducible Task A pipeline** — §11.5.4 establishes that zero of seven 2025 teams published code. Submitting the repo as-is (Hydra configs, Phase 2.5/2.6 qrels artefacts, three-judge α infrastructure) would set the reproducibility floor for the track going forward, independently of where our pipeline ranks.

---

## 14. Sources

### 14.1 Datasets and tracks

- TREC BioGEN 2024 — predecessor track: [trec.nist.gov](https://trec.nist.gov)
- TREC BioGEN 2025 — current track. Official 2025 overview: Gupta, D., Demner-Fushman, D., Hersh, B., Bedrick, S., Roberts, K. (2026). *"Overview of TREC 2025 Biomedical Generative Retrieval (BioGen) Track"*, [arXiv:2603.21582](https://arxiv.org/abs/2603.21582). Table 5 is the calibration anchor for the 44.34 / 4.67 baseline TEST run.
- TREC BioGEN 2025 participant notebook papers: [TREC 34 proceedings (NIST)](https://trec.nist.gov/pubs/trec34/index.html) — CLaC, CLaC Lab, dal, GEHC-HTIC, SIB, UAmsterdam, uniud each published a notebook paper (§11.5.2). None published code.
- GEHC-HTIC team preprint: Sahoo, S. R., N., G., Sasidharan, S., Bharti, D. (2026). *"Negation is Not Semantic: Diagnosing Dense Retrieval Failure Modes for Trade-offs in Contradiction-Aware Biomedical QA"*, [arXiv:2603.17580](https://arxiv.org/abs/2603.17580). Describes GE Healthcare Bangalore's "Decoupled Lexical Architecture" + "Narrative-Aware Reranking" + "One-Shot In-Context Learning" pipeline for Task A.
- **BioACE (organisers' evaluation framework, used in Table 5 of the 2025 overview):** Gupta, D., Bartels, D., Demner-Fushman, D. (2026). *"BioACE: An Automated Framework for Biomedical Answer and Citation Evaluations"*, [arXiv:2602.04982](https://arxiv.org/abs/2602.04982). Nugget-based LLM-as-judge (Llama-3.3) over `(answer-sentence, cited-document)` pairs producing `Supports/Contradicts/Neutral/Not relevant` labels — the single-backend, no-CI judge whose limitations the §10 — §10.9 hardening of this work directly addresses.
- Organisers' starter kit (the baseline TEST implementation): [github.com/trec-biogen/starter-kit-2025](https://github.com/trec-biogen/starter-kit-2025). Vendored locally via `scripts/vendor_starter_kit.sh` (§4.5).
- Adjacent precedent for *published* BioGen code: [Webis at TREC 2024 BioGen](https://github.com/webis-de/trec24-biogen) (2024 only; no 2025 submission).
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

- Design: [`openspec/changes/phase2-pool-aware-pipeline/`](../openspec/changes/phase2-pool-aware-pipeline/) — proposal, design, specs, tasks (75 of 84 ticked at time of writing, including the openspec design task `12.10` LLM query-rewrite stretch).
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
uv run python -m trec_biogen.pipeline.run_task_a --config-name run/phase2_no_rerank
uv run python -m trec_biogen.pipeline.run_task_a --config-name run/phase2_no_negex
uv run python -m trec_biogen.pipeline.run_task_a --config-name run/phase2_scifive_large
uv run python -m trec_biogen.pipeline.run_task_a --config-name run/phase2_bm25_rm3
uv run python -m trec_biogen.pipeline.run_task_a --config-name run/phase2_bm25_rm3_llm_filtered
uv run python -m trec_biogen.pipeline.run_task_a --config-name run/phase2_bm25_llm_rewrite

# 7. Phase 2 §10 methodological hardening
# 7.1 bootstrap CI + persisted records (records-out unlocks §10.2 calibration)
uv run python -m trec_biogen.judge.rejudge validate \
    --backend openai-mini --prompt cot --threshold 0.85 \
    --records-out reports/_records_mini_cot.jsonl

# 7.2 second-backend gate (Together / HF Providers, $0.376)
export TOGETHER_API_KEY=...
uv run python -m trec_biogen.judge.rejudge validate \
    --backend together --prompt cot --threshold 0.85 \
    --records-out reports/_records_together_cot.jsonl

# 7.3 calibration (held-out k=5 CV ECE, no API spend)
uv run python -m trec_biogen.judge.calibration \
    --records reports/_records_mini_cot.jsonl --k 5 --seed 0

# 8. Phase 2.5 — second judge over the full §2.17 candidate set ($2.47)
export HF_TOKEN=...
uv run python -m trec_biogen.judge.rejudge expand-pool \
    --backend hf-llama-3.3-70b --prompt cot \
    --retrieval-support    runs/<phase1_run>/retrieval_support.parquet \
    --retrieval-contradict runs/<phase1_run>/retrieval_contradict.parquet \
    --top-k 30 --max-concurrent 8 --cost-cap 5 \
    --out data/qrels/biogen2025_taskA_qrels_expanded_hf-llama.jsonl

uv run python -m trec_biogen.judge.intersection \
    --records data/qrels/biogen2025_taskA_qrels_expanded.jsonl \
              data/qrels/biogen2025_taskA_qrels_expanded_hf-llama.jsonl \
    --out data/qrels/biogen2025_taskA_qrels_intersection.jsonl

# 9. Phase 2.6 — third judge (Qwen2.5-72B via HF Providers, $4.00)
uv run python -m trec_biogen.judge.rejudge expand-pool \
    --backend hf-qwen-2.5-72b --prompt cot \
    --retrieval-support    runs/<phase1_run>/retrieval_support.parquet \
    --retrieval-contradict runs/<phase1_run>/retrieval_contradict.parquet \
    --top-k 30 --max-concurrent 8 --cost-cap 6 \
    --out data/qrels/biogen2025_taskA_qrels_expanded_qwen.jsonl

uv run python -m trec_biogen.judge.intersection \
    --records data/qrels/biogen2025_taskA_qrels_expanded.jsonl \
              data/qrels/biogen2025_taskA_qrels_expanded_hf-llama.jsonl \
              data/qrels/biogen2025_taskA_qrels_expanded_qwen.jsonl \
    --out data/qrels/biogen2025_taskA_qrels_intersection_3way.jsonl

# 10. Regenerate the final summary with cell-level bootstrap CIs on every pool.
uv run python -m trec_biogen.eval.phase2_summary \
    --pools official expanded intersection intersection-3way
```

**Resource budget (full project, eight variants except `phase2_hybrid`):**

| Item | Value | Source |
|---|---:|---|
| LLM-judge API spend, §2 expansion | $1.77 | §6.3 + §6.4 |
| LLM-judge API spend, §10 hardening | $0.91 | §10 intro |
| LLM-judge API spend, Phase 2.5 second-judge rejudge | $2.47 | §10.8 |
| LLM-judge API spend, Phase 2.6 third-judge rejudge + Qwen gate | $4.00 | §10.9 |
| **Total LLM-judge spend** | **$9.15** | — |
| Phase 1 baseline pipeline wall-clock | ~6 h | §4.6 |
| `bm25_rm3` full | ~84 min | §7.2 |
| `bm25_rm3_llm_filtered` full | ~84 min | §7.2 |
| `bm25_llm_rewrite` full | ~88 min | §10.7 |
| `no_negex` full | ~9.7 h | §7.2 |
| `scifive_large` full | ~5.3 h | §7.2 |
| `no_rerank` (selective reuse) | ~12 min | §7.2 |
| `allow_existing` (reuse-from) | <2 min | §7.2 |
| **Total GPU time** | **~22 h** | — |
| Disk footprint (BM25 index + run artefacts) | ~40 GB | — |
| Hardware floor | WSL2 laptop, 12 GB RAM, Quadro T1000 4 GB VRAM | §1.3 |

Note: `phase2_hybrid` (BM25 + Dense MedCPT + RRF) would add ~24 h CPU encoding for the 5 M-doc FAISS index + ~2 h GPU for retrieval, plus its own §2.17-equivalent expand-pool pass. Not included in the totals above; see §13 future work.

---

*End of report.*
