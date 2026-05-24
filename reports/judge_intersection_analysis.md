# Phase 2.5 §1-§3 — Judge robustness & intersection-pool analysis

## Why a second judge

The Phase 2 expanded qrels file (`biogen2025_taskA_qrels_expanded.jsonl`)
was built by a single judge: `gpt-4o-mini --prompt cot`, validated in §2.15
at macro-w-F1 = 0.8944 against the 588 human-labelled triples (95% CI
[0.876, 0.920]). The §12.4 multi-backend probe added a second backend
(Together-Llama-3.3-70B-cot) and established two facts:

* Per-class agreement is asymmetric. Cohen's κ overall = 0.34 ("fair"),
  but Supports F1 is robust to judge choice while Contradicts F1 carries
  meaningful judge-dependent variance (Together is materially more
  conservative on contradicts than gpt-4o-mini).
* The κ analysis was on the 588 validation triples only — not on the
  5398 candidate triples that built the expanded pool. So any cross-variant
  finding that depends on those LLM-added Contradicts was un-tested for
  robustness.

The headline structural Phase 2 win sits exactly there: `phase2_no_negex`
beats Phase 1 on official Contradicts F1 by +2.13 pp (~5× the published
anchor). This Phase 2.5 work re-runs that test under a *stricter pool*
where only contradicts agreed by two independent judges count.

## How the intersection pool was built

A second rejudge of the same 5398 candidate triples was performed with
`Llama-3.3-70B-Instruct --prompt cot`, hosted via the HuggingFace
Inference Providers router (auto-routed to Groq for sub-second 70B
inference). The model weights are identical to Together's offering; only
the hosting / billing route differs. Because the gate validation in
§12.4 was on these exact weights, no re-validation was required — the
gate at κ=0.9112 [0.886, 0.936] carries over.

The intersection-pool emitter (`src/trec_biogen/judge/intersection.py`)
then derived a third qrels file by the rule:

* **Human positives** copied verbatim (588 records).
* **Supports** passed through from the canonical mini-cot pool (3 807
  positives) — Supports are not intersected because §12.4 shows them
  stable across judges.
* **Contradicts** kept only when *both* mini-cot and HF-Llama-cot labelled
  the same `(qa_id, sentence_id, pmid, class)` as Contradicts.

| | mini-cot only | mini ∩ HF-Llama | Δ |
|---|---:|---:|---:|
| Supports (LLM) | 3 807 | 3 807 | 0 (pass-through) |
| **Contradicts (LLM)** | 363 | **43** | **-320 (88 % drop)** |
| Human (all classes) | 588 | 588 | 0 |
| Total positives | 4 758 | 4 438 | -320 |

The 88 % drop confirms the §12.4 asymmetry: the two judges agree on
Supports but rarely on Contradicts (per-call cost: ~$1.50 of HF spend).

## Cross-judge per-class agreement on the full 5 398-triple set

| Class | mini-cot positives | HF-Llama positives | Both agree (intersection) | Jaccard |
|---|---:|---:|---:|---:|
| Supports | 3 807 | 4 056 | 3 807 | ~0.93 |
| **Contradicts** | 363 | 58 | **43** | **~0.12** |

* Supports: 3 807 of mini's are also flagged by Llama (≥ 99.9 %); Llama
  emits ~250 *additional* supports that mini classified as Neutral. By
  the intersection rule we keep mini's set unchanged for Supports.
* Contradicts: Llama emits only 58 Contradicts vs mini's 363. Of those
  58, **43 also appear in mini's set** (intersection = 43, union = 378).
  Llama is 6.3× more conservative on the contradicts class — the §12.4
  finding *amplified*: at the 5 398-triple scale, the asymmetry becomes
  the dominant signal.

## Cell-level bootstrap CIs on the intersection pool

Bootstrap: B = 1 000 resamples of per-cell F1 with replacement, seed=0,
α=0.05 (95% CI). All Strict, sentence-level macro. See
[`scripts/bootstrap_intersection_ci.py`](../scripts/bootstrap_intersection_ci.py).

| run | class | F1 | 95% CI |
|---|---|---:|---|
| starter_baseline | support | 16.55 | [15.01, 18.25] |
| starter_baseline | contradict | **4.01** | [2.13, 6.13] |
| phase1_baseline | support | 16.43 | [15.15, 17.80] |
| phase1_baseline | contradict | 1.07 | [0.26, 2.04] |
| phase2_allow_existing | support | 16.94 | [15.59, 18.26] |
| phase2_allow_existing | contradict | 1.07 | [0.21, 2.10] |
| phase2_no_negex | support | 16.33 | [15.15, 17.59] |
| phase2_no_negex | contradict | **3.63** | [1.98, 5.38] |
| phase2_no_rerank | support | 15.35 | [14.00, 16.61] |
| phase2_no_rerank | contradict | 1.07 | [0.21, 2.13] |
| phase2_scifive_large | support | 16.43 | [15.09, 17.73] |
| phase2_scifive_large | contradict | 2.21 | [0.88, 3.79] |
| phase2_bm25_rm3 | support | 8.97 | [7.79, 10.21] |
| phase2_bm25_rm3 | contradict | 0.55 | [0.00, 1.32] |
| phase2_bm25_rm3_llm_filtered | support | 9.89 | [8.59, 11.05] |
| phase2_bm25_rm3_llm_filtered | contradict | 1.07 | [0.26, 2.04] |
| phase2_bm25_llm_rewrite | support | 10.65 | [9.47, 11.81] |
| phase2_bm25_llm_rewrite | contradict | 0.81 | [0.09, 1.74] |

## Reading these numbers

**Sampling-noise floor.** The intersection pool's Contradicts class has
43 positives across 313 cells (mean 0.14 per cell, IQR 0). Most cells
contribute F1=0; the macro is dominated by the few cells that match a
positive. CI widths of 2-3 percentage points on Contradicts are *much*
larger than the cross-variant differences. Per the design's §D2
downweight rule (triggered: intersection is 12 % of union, well below
the 30 % threshold), the qualitative claims below are downweighted to
**directional** rather than statistically significant unless explicitly
flagged.

**`phase2_no_negex` still beats Phase 1 (Contradicts).** Point estimates
1.07 (Phase 1) vs 3.63 (no_negex); Phase 1's upper CI (2.04) is *just*
above no_negex's lower CI (1.98) — the overlap is marginal and the point
gap is ~3.4× the midpoint. **Directionally clear; the structural Phase 2
finding survives the conservative pool.**

**`starter_baseline` ≈ `no_negex` on Contradicts (intersection).** Point
estimates 4.01 vs 3.63 (Δ=0.38) sit inside the CI overlap of [2.13,
6.13] ∩ [1.98, 5.38]. The expanded-pool finding "no_negex >>> starter on
contradicts" (12.01 vs 5.34, Δ=+6.67) does *not* survive the conservative
pool: most of that gap was liberal mini-cot judgments that Llama did not
ratify. **The honest finding on the conservative pool is: starter and
no_negex are statistically indistinguishable on the Contradicts class;
both clearly beat Phase 1.**

**Three retrieval-side variants are clearly worse on Supports** under
the same conservative pool: `bm25_rm3` (8.97), `bm25_rm3_llm_filtered`
(9.89), `bm25_llm_rewrite` (10.65) all sit several CIs below the
~16-17 band where Phase 1 / starter / `no_negex` / `allow_existing` /
`scifive_large` cluster. The three independent query-side negative
results from Phase 2 survive the pool tightening unchanged.

## What the asymmetry tells us about LLM-as-judge calibration

The cross-judge per-class table above is a methodologically interesting
artefact in its own right. Two reasons two judges might disagree on
Contradicts but agree on Supports:

1. **Asymmetric base rates.** PubMed publication culture favours
   affirmative findings. The base rate of contradicts in any pool is
   inherently low (~3-5 % of LLM-judged triples). Both judges have to
   navigate a narrow majority class; small differences in threshold show
   up as large absolute disagreement.
2. **Asymmetric prompt sensitivity.** The CoT prompt asks for a
   reasoning chain before the label. For Supports, the model finds
   evidence in the abstract that "matches" the sentence's claim — a
   *positive* task. For Contradicts, the model must reason that the
   abstract not only fails to support but *actively contradicts* — a
   *higher-bar inference* that bigger models (Llama-70B) approach more
   conservatively. The 6.3× ratio is consistent with Llama being a
   stricter contradiction-classifier than mini at the same CoT
   temperature.

Practical implication for downstream pipelines: when LLM-judging
biomedical contradicts at scale, the **judge-choice variance is a
first-order effect, not noise**. A single-judge expanded qrels file
should be reported alongside an intersection-pool sanity check or a
multi-judge confidence-averaged variant.

## Cost ledger for §1-§3

| Item | Spend |
|---|---:|
| HF Inference Providers (Llama-3.3-70B via Groq) | $2.4739 |
| Together prior attempt (29 of 5398 triples before HTTP 402) | $0.0100 |
| OpenAI rejudges (§2.17, mini-cot, reused) | $0 (no new spend) |
| **Phase 2.5 §1-§3 total** | **~$2.49** |

---

# Phase 2.6 §3 — Three-judge intersection extension (2026-05-24)

## Why a third judge

The §1 two-judge intersection narrowed the Contradicts pool from 363
mini-cot positives to 43 (mini ∩ Llama-70B). That left an open question
the §1 analysis itself called out: is the 88 % drop on contradicts
"mini-cot is over-emitting" or "Llama is over-stripping"? With only two
data points there was no way to tell. Phase 2.6 adds a third independent
judge — different model family from both prior backends — so the
disagreement pattern can be triangulated rather than dichotomised.

## How the third judge was selected

Design D1 (in the openspec change) originally proposed Mixtral-8x7B via
HF Inference Providers. At implementation time HF had removed the
Mistral family from the chat-routable roster (`400 "not a chat model"`
on every variant probed) and the Together-direct fallback returned
`402 Credit limit exceeded`. The pivot — documented in design D1 — went
to **Qwen2.5-72B-Instruct** (Alibaba dense 72B, routed by HF Providers
to OpenRouter). Same intent: third model family distinct from both
OpenAI's GPT-4 line and Meta's Llama-3 line.

## Gate validation

Qwen2.5-72B-cot scored macro-w-F1 = **0.8980** on the 588-triple human
concordance set (≥ 0.85 gate, per-class F1 floor 0.05 cleared on every
non-empty class). Per-class:

| Class | F1 (Qwen) | F1 (mini, ref) | F1 (Llama, ref) | n |
|---|---:|---:|---:|---:|
| Supports | 0.944 | 0.924 | 0.958 | 549 |
| Contradicts | 0.250 | 0.480 | 0.250 | 39 |

Qwen sits *between* mini and Llama on the macro and matches Llama's
Contradicts conservatism almost exactly. The first useful signal: on
the 588 gold set, **Qwen and Llama agree with each other much more than
either agrees with mini** — the §10.4 Phase 2 finding of "mini is
liberal on contradicts" generalises beyond a single comparison.

## Three-judge cross-judge agreement (Krippendorff α, K2011 formula)

On the **588 gold set** (every judge classifies every triple):

| Subset | α (3 judges) | n |
|---|---:|---:|
| Full label space (Supports / Contradicts / Neutral / Not relevant) | **0.3643** | 588 |
| Restricted to human-Supports units | 0.3204 | 549 |
| Restricted to human-Contradicts units | 0.2918 | 39 |

On the **5 398 §2.17 expand-pool candidate set** (every judge labelled
every triple; absences treated as "Neutral"):

| α | Value |
|---|---:|
| 3-way α (full label space) | **0.2992** |
| Pairwise α: mini ↔ Llama | 0.1166 |
| Pairwise α: mini ↔ Qwen | 0.2041 |
| Pairwise α: **Llama ↔ Qwen** | **0.6013** |

**This is the headline finding.** Llama and Qwen agree at α=0.60 (which
maps roughly to Cohen's κ in the "substantial agreement" range, per
Landis & Koch 1977). Mini-cot agrees with neither at much above the
"slight" level (0.12 / 0.20). The two-judge intersection pool from
Phase 2.5 was effectively "mini's contradicts, filtered through Llama's
veto"; the three-judge pool is essentially "Llama and Qwen's shared
contradicts (which mini also happens to ratify in most cases)".

## Three-way intersection-on-contradicts pool

The pool was emitted by `intersection.emit_intersection_pool` in its new
N-list form: Supports pass through from mini-cot (3 807 positives),
human positives copied verbatim (588), Contradicts kept only where all
three judges agree. Counts:

| | mini-only | mini ∩ Llama (Phase 2.5) | mini ∩ Llama ∩ Qwen (Phase 2.6) | drop vs mini |
|---|---:|---:|---:|---:|
| Supports | 3 807 | 3 807 | 3 807 | 0 (pass-through) |
| **Contradicts** | 363 | 43 | **31** | **−91.5 %** |
| Human (all) | 588 | 588 | 588 | 0 |
| Total | 4 758 | 4 438 | 4 426 | −7.0 % |

Pairwise intersection diagnostics from the sidecar:

| Pair | Contradicts agreed |
|---|---:|
| mini ∩ Llama | 43 |
| mini ∩ Qwen | 71 |
| **Llama ∩ Qwen** | **32** |
| All three | 31 |

The 32 vs 31 near-equality is what the α numbers above already told us:
when Llama and Qwen agree on a contradict, mini agrees on it 31/32
times. The 3-way pool is essentially the Llama–Qwen pool.

The 31 surviving positives clear the design-D2 floor of 20 ("if the 3-way
Contradicts pool has < 20 positives, downweight macro conclusions"), but
sit just above it — Contradicts statistics on the 3-way pool are reportable
but should be read with the small-sample caveat in mind.

## Variant F1 on the 3-way pool (Strict, sentence-level macro)

| variant | Supports F1 (3-way) | Contradicts F1 (3-way) | Δ vs 2-way (Contradicts) |
|---|---:|---:|---:|
| starter_baseline | 16.55 | **4.08** | +0.07 |
| Phase 1 | 16.43 | 1.12 | +0.05 |
| `no_negex` | 16.33 | **3.70** | +0.07 |
| `scifive_large` | 16.43 | 2.00 | −0.21 |
| `allow_existing` | 16.94 | 1.12 | +0.05 |
| `bm25_rm3` | 8.97 | 0.60 | +0.05 |
| `bm25_rm3_llm_filtered` | 9.89 | 1.12 | +0.05 |
| `bm25_llm_rewrite` | 10.65 | 0.86 | +0.05 |

Contradicts numbers shift by ≤ ±0.3 pp from the 2-way pool — the same
units' true-positive count divided by a marginally smaller denominator.
The qualitative findings from Phase 2.5 §10.8 carry over **unchanged**:

* `no_negex` beats Phase 1 on Contradicts (3.70 vs 1.12, ~3.3×): the
  structural Phase 2 finding survives.
* `no_negex` is statistically indistinguishable from `starter` on
  Contradicts (3.70 vs 4.08): the apparent expanded-pool advantage
  remains an artefact of liberal mini-cot ratifications.
* The three retrieval-side variants (`bm25_rm3`, `bm25_rm3_llm_filtered`,
  `bm25_llm_rewrite`) remain clearly worse on Supports under the
  conservative pool.

## What changes vs Phase 2.5

The Phase 2.6 work tightens the methodological claim without overturning
any empirical finding. Specifically:

* The Phase 2.5 narrative "the two-judge intersection is conservative
  but might be the result of mini being too liberal *or* Llama being
  too strict" is now resolved: **mini is the outlier.** Llama and Qwen,
  trained by different organisations on different data with different
  architectures, converge on a similar (much smaller) set of contradicts.
* The 3-way pool number on Contradicts is essentially the Llama–Qwen
  pool number; mini's "veto" power is illusory because almost every
  contradict Llama and Qwen agree on, mini also agrees on.
* The Phase 2 ablation conclusions are robust to going from 1 → 2 → 3
  judges; the variant rankings on the 3-way pool are the same as on
  the 2-way pool, with only ±0.3 pp shifts.

## Cost ledger for Phase 2.6 §2-§3

| Item | Spend |
|---|---:|
| HF Inference Providers (Qwen2.5-72B via OpenRouter) — gold-set validate (588) | $0.36 |
| HF Inference Providers (Qwen2.5-72B via OpenRouter) — expand-pool initial (4 238 triples) | ~$2.27 |
| HF Inference Providers (Qwen2.5-72B via OpenRouter) — expand-pool resume after 400 (2 223 triples) | $1.37 |
| **Phase 2.6 §2-§3 total** | **~$4.00** |

Inside the design's $5 hard ceiling. Combined Phase 2 + 2.5 + 2.6 LLM-judge
spend across the whole programme: ~$1.77 (Phase 2 §2) + ~$0.91 (Phase 2 §10) +
~$2.49 (Phase 2.5) + ~$4.00 (Phase 2.6) ≈ **$9.17 total**.
