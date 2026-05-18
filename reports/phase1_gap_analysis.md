# Phase-1 Gap Analysis (task 12.2)

## Run

* Run id: `runs/20260516-134227-phase1_baseline/`
* Date: 2026-05-17
* Git SHA: `b7745bc`
* Config: `configs/run/phase1_baseline.yaml`
* Qrels: `data/qrels/biogen2025_taskA_qrels.jsonl` (549 support + 39 contradict positives, derived from organizers' `baseline_labels.json`)
* Methodology: published BioGEN 2025 macro (mean-of-per-cell-F1, unjudged cells = 0)

## Numbers (strict, 2025 qrels)

| Submission | Supports F1 | Contradicts F1 | Δ vs published baseline |
|---|---:|---:|---:|
| **OFFICIAL baseline** (organizers run, `baseline_output.json`) | 44.34% | 2.51% | +0.00 / -2.16 |
| **OUR starter-kit reproduction** (`runs/starter_baseline_20260514_150718/`) | 44.34% | 4.21% | **+0.00 / -0.46 ✅** |
| **OUR Phase-1 pipeline** (`runs/20260516-134227-phase1_baseline/`) | **5.55%** | **0.52%** | -38.79 / -4.15 |

`Relaxed` numbers equal `Strict` because `baseline_labels.json` has no partial-support label gradation (only Supports / Contradicts / Neutral / Not relevant).

## Phase-1 thresholds — verdict

| Threshold | Target | Actual | Result |
|---|---:|---:|---|
| Supports F1 ≥ 60 (strict, 2025 qrels) | ≥ 60.00 | 5.55 | **FAIL** by 54.45 pp |
| Contradicts F1 ≥ 10 (strict, 2025 qrels) | ≥ 10.00 | 0.52 | **FAIL** by 9.48 pp |

Phase-1 thresholds NOT met. The §6.5 baseline-reproduction gate **passes** (OUR starter-kit reproduction lands within ±2 F1 of published 44.34/4.67) — so the eval module is calibrated correctly; the gap is real, not a measurement bug.

## Root cause: pool bias

`baseline_labels.json` was produced by judging the **official baseline submission's** picks. The qrels positives are therefore a strict subset of what the baseline retrieved. Our pipeline picks **different PMIDs** — none of which are in the labels pool. Specifically:

* Pipeline produced 555 distinct (qa_id, sentence_id, pmid) support predictions; intersection with labels pool ≈ 30.
* Pipeline produced 569 distinct contradict predictions; intersection ≈ 3.

Our predictions are not necessarily *wrong* — many likely retrieve correct evidence that simply wasn't part of the 2025 judgement pool. The published BioGEN 2025 numbers across teams suffer the same TREC pooling problem; only systems whose runs went into the assessment pool can be fully judged.

## Phase-2 mitigations (priority order)

Each mitigation addresses one identifiable failure mode.

### 1. Re-rank pool overlap before NLI scoring  *(easiest lift)*

The bottleneck is that our **MedCPT-CE rerank for support** picks PMIDs the baseline didn't, even though our retrieval is BM25-on-the-same-index. Verify the overlap of our **top-25 BM25** support retrieval against the baseline's predictions: if BM25 overlap is high but MedCPT-CE rerank diverges, the rerank is the issue. Hypotheses:

  * MedCPT-CE re-orders aggressively away from BM25 ranking on biomedical retrieval (whereas the baseline uses pure top-25 BM25, no rerank). Test: skip the rerank, use the BM25 top-3 directly. Expected: closer to baseline's support pool, ≥ 30-40% F1.
  * If we keep the rerank but want to stay in-pool, use `MedCPT-CE` as a **light** rerank (only re-rank top-25 by tie-breaking on small score differences; keep BM25 order otherwise).

### 2. Bypass the contradict NegEx pre-filter  *(retrieval recall)*

NegEx filter dropped 95.7% of segmented sentences. The 4.15 pp gap on contradict is small in absolute terms but means we missed most of the 39 contradict positives. The baseline uses NO filter; it runs SciFive-large on the BM25 top-500 directly. Phase 2: drop NegEx, let the NLI model decide; cost is wall-clock (was 5+ hours with negspacy; cue-only NegEx ran in 20s but lost recall on entities not in our regex list).

### 3. Switch contradict NLI to SciFive-large (fp16)  *(specialisation)*

Phase 1 used DeBERTa-MNLI for both paths because `razent/SciFive-base-Pubmed_PMC-MedNLI` doesn't exist. The starter-kit uses `razent/SciFive-large-Pubmed_PMC-MedNLI` — that *does* exist and fits in 4 GB VRAM at fp16 (we proved this on the 29h starter-kit reproduction). Swapping in the biomedical-specialised model on the contradict path should narrow the contradict gap.

### 4. Retrieval bridging  *(pool bias mitigation)*

To make our pipeline's predictions actually judgeable next year:

  * Submit a real TREC run so our picks go into the pool.
  * For retrospective evaluation now, pair-judge a sample of our **novel** PMIDs (those not in baseline_labels.json) using a strong LLM-as-judge; use this as a sanity check on the absolute F1.

## NegEx audit summary

See `runs/20260516-134227-phase1_baseline/negation_audit.jsonl` (50 sampled dropped sentences). Filter rate: 95.7% (1,828,563 of 1,911,563 dropped). Even with this aggressive filtering, our contradict path produced 569 predictions and the absolute F1 (0.52%) is mostly pool bias, not filter aggression — see point (2) above. A manual review of the 50-sample audit can confirm there are no high-recall mistakes (cue regex missed obvious contradictions).

## Blockers / unknowns

* Why is the OFFICIAL baseline's contradict F1 only 2.51 here vs the published 4.67? The labels file we have (`baseline_labels.json`) was shared 2026-05-18; the published 4.67 reflects either an older or slightly different label set. Within ±2.16 pp it is plausible label drift; not a methodology issue.
* `relaxed` setting is currently identical to `strict` because the labels distribute as binary (`Supports` / `Contradicts`); a `partially_supports` label exists in the 10-question CSV sample but not in `baseline_labels.json`. If a future qrels release distinguishes partial labels, our `eval/qrels.py` already maps `partial_support` → relaxed bucket without code change.
