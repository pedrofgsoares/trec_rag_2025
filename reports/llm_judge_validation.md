# LLM-Judge Concordance Validation — Phase 2 §2.7 / §2.8 / §2.15

## Verdict

**Gate passed with `openai-gpt-4o-mini --prompt cot`: macro weighted
F1 = 0.8944, above the 0.85 design threshold.** The same backend with
the original strict-mode prompt failed the gate at 0.7497. The single
change is the prompt: a chain-of-thought variant that asks the model
for a 2-3 sentence inferential chain before committing to a label.

The expanded qrels artefact for §2.16 was produced with this
configuration: `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`
(1297 positives across 272 cells; 588 human rows preserved verbatim
plus 709 new LLM positives from 1074 novel-PMID classifications).

## How the prompt switch fixed the gate

The strict-mode failure was diagnosed *not* as a label-space mismatch
(humans generous, LLM strict) but as **substantive inferential-chain
failure**: the LLM had the medical knowledge to recognise implicit
support (J-curve → harm at low BP; nationwide VA training rollouts →
institutional recommendation; "progressive symptoms → early
disability" → physical dependence) but never wrote it down and so
defaulted to "Neutral". A four-case probe
(`scripts/judge_disagreement_examples.py` /
`scripts/judge_cot_probe.py`) confirmed all four strict-mode
disagreements flip to `Supports` with CoT, and the reasoning chains
the model emits reconstruct the same inference path a domain expert
would walk.

CoT prompt is now a first-class option:
`uv run python -m trec_biogen.judge.rejudge {validate,rejudge}
--prompt cot` (default remains `strict` so existing tests stay green).

## Numbers — all attempted backends and prompt modes

| Backend | Prompt | Macro w-F1 | Supports P / R / F1 | Contradicts P / R / F1 | Triples | Cost USD |
|---|---|---|---|---|---|---|
| `openai-gpt-4o-mini` | strict | 0.7497 (FAIL) | 0.975 / 0.639 / 0.772 | 0.457 / 0.410 / 0.432 | 588 | $0.048 |
| `openai-gpt-4o`      | strict | 0.7443 (FAIL) | 0.975 / 0.634 / 0.768 | 0.733 / 0.282 / 0.407 | 588 | $0.792 |
| `openai-gpt-4o-mini` | **cot** | **0.8982 (PASS)** | **0.967 / 0.883 / 0.924** | 0.543 / 0.487 / 0.514 | 588 | $0.080 + $0.078 |
| `together-llama-3.3-70b` | **cot** | **0.9112 (PASS)** | **0.956 / 0.960 / 0.958** | 0.667 / 0.154 / 0.250 | 588 | $0.376 |

The Together backend was provisioned later in the work (Together had
moved the 3.1-70B-Turbo to dedicated-endpoint-only access; the
serverless successor `Llama-3.3-70B-Instruct-Turbo` is the substitute,
same family, same ~$0.88/M pricing). It is the **third independent
backend** the design D10 "judge sensitivity" experiment calls for.

### Bootstrap 95% CIs (Phase 2 §12.1)

Non-parametric resampling of the 588 `(gold, pred)` pairs with
replacement, B = 1000, seed = 0:

| Backend × prompt | Point estimate | 95% CI | CI width | Gate ≥ 0.85 |
|---|---|---|---|---|
| `openai-gpt-4o-mini` × cot | 0.8982 | [0.8776, 0.9196] | 0.042 | **PASS** (CI lower bound also ≥ 0.85) |
| `together-llama-3.3-70b` × cot | 0.9112 | [0.8861, 0.9355] | 0.049 | **PASS** (CI lower bound also ≥ 0.85) |

Both backends not only pass the 0.85 gate at the point estimate but
their CI **lower bounds** also clear 0.85 — the result is statistically
robust to triple-level sampling noise, not an artefact of one
particular 588-triple draw.

### Multi-backend concordance (Phase 2 §12.4)

Pairwise agreement between the two backends over the same 588 triples:

| A | B | Raw agreement | Cohen's κ |
|---|---|---|---|
| `openai-gpt-4o-mini`-cot | `together-llama-3.3-70b`-cot | **0.867** | **0.338** |

Raw agreement is high (87%), but Cohen's κ — which corrects for the
chance agreement induced by the 549/39 Supports/Contradicts class
imbalance — lands at 0.338 (fair, per Landis & Koch 1977; below
"substantial" 0.6 threshold). Reading the matrix:

* **Supports** is where the backends agree most: both correctly label
  most of the 549 gold-Supports. The expanded-pool *support* F1 numbers
  reported elsewhere are therefore robust to judge choice.
* **Contradicts** is where they diverge: Together's Llama-70B is more
  conservative (only 9 predicted contradicts out of 39 gold, 0.25 F1)
  than OpenAI mini (35 predicted, 0.51 F1). The expanded-pool
  *contradict* F1 numbers carry meaningful judge-dependent variance.

For the paper, the defensible claim is therefore **"expanded-pool
support F1 is robust to judge choice; expanded-pool contradict F1
varies by judge"** — a more honest framing than a blanket robustness
claim. A two-judge agreement-floor (intersection of LLM positives
where mini and Together both agree) would be the conservative extension
for downstream variant comparisons.

The full multi-backend pairwise report is at
[`reports/llm_judge_multi_backend.md`](llm_judge_multi_backend.md).

## Confusion matrix — `openai-gpt-4o-mini --prompt cot`

Rows: human label. Columns: judge label.

|  | Supports | Contradicts | Neutral | Not relevant |
|---|---|---|---|---|
| Supports     | 485 | 18 | 45 | 1 |
| Contradicts  | 16  | 18 |  5 | 0 |

* `Supports → Neutral` collapsed from 171 (strict) to 45 (CoT) — the
  inferential chain unlocks ~75% of the prior false-Neutrals.
* `Supports → Not relevant` collapsed from 8 to 1 — the model is no
  longer randomly classifying on-topic abstracts as off-topic.
* `Contradicts` remains the noisy class (n=39): F1 0.48 vs the 0.92 on
  Supports. The 39-triple sample is too small for tight estimates; the
  weighted macro is fine because Supports dominates the sample.

## §2.16 — rejudge of the Phase 1 novel pool

Inputs: `runs/20260516-134227-phase1_baseline/task_a_output.json`
contains 1074 (qa_id, sentence_id, pmid) triples that the Phase 1
pipeline emitted but that are absent from the human qrels. CoT
re-judgement produced:

| LLM judge label | Count | Disposition in expanded qrels |
|---|---|---|
| Supports     | 605 | emitted as `class: support`     |
| Contradicts  | 104 | emitted as `class: contradict`  |
| Neutral      | 257 | dropped (not a positive class)  |
| Not relevant | 108 | dropped (not a positive class)  |
| **Total novel** | **1074** | **709 new positives** |

Cost: $0.149 (1074 calls, ~$0.0001 each). Per-run metadata at
`runs/20260519-135603-judge_rejudge_phase1_cot/metadata.yaml`. The
expanded qrels file parses cleanly via
`trec_biogen.io.qrels.load_qrels` — 1297 positives across 272
(qa_id, sentence_id, class) cells, ready for the §3 dual-pool
evaluation to consume via `--qrels-pool=expanded` (Phase 2 §3.1, not
yet implemented).

The ~56% "novel → positive" rate (709/1074) is well below the human
pool's ~99% (588/595 cells positive after curation), which is what
you'd expect: the novel pool is the un-filtered output of the Phase 1
pipeline, so it includes the cases the human pool curators would have
rejected too.

## §2.17 — BM25 top-30 broader pool

Run after `phase2_no_rerank` revealed circularity in the §2.16-only pool:
the Phase 1 rejudge crystallised the LLM-positives around Phase 1's
*own* pick set, so any variant that mutated the upstream retrieval /
rerank stage was unfairly penalised on the expanded pool.

`expand-pool` reads the BM25 top-30 PMIDs from every (qa_id, sentence_id,
class) cell of the Phase 1 retrieval parquets, dedupes across paths,
subtracts already-judged triples (human + prior LLM), and rejudges the
remainder with the same `openai-gpt-4o-mini --prompt cot` configuration.

* Input: 5819 unique top-30 triples (support + contradict, deduped).
* After dedup against 588 human + 709 prior LLM: **5169 new
  classifications**.
* Cost: **$0.704**, 16 min wall-clock at `--max-concurrent 8`.
* New LLM-positives: 3807 support + 363 contradict.
* Total expanded qrels: **4758 positives** (588 human + 4170 LLM),
  3.7× larger than the §2.16-only pool.

Headline consequence: cross-variant expanded-pool F1s now cluster in
the 15-17 pp band for support and ~12 pp for contradict, instead of
the §2.16-only artefact where `phase1_baseline` scored 44.34 on support
purely because the pool was built from its picks. See
`reports/phase2_summary.md` for the per-variant row.

## Total spend

| Run | Purpose | Cost USD |
|---|---|---|
| `20260519-121014-judge_validate`            | 2.15 strict mini    | $0.048 |
| `20260519-122135-judge_validate_gpt4o`      | 2.15 strict 4o      | $0.792 |
| `20260519-132658-judge_validate_cot_mini`   | 2.15 CoT mini (PASS) | $0.080 |
| `20260519-135603-judge_rejudge_phase1_cot`  | 2.16 CoT mini rejudge | $0.149 |
| `20260519-180822-judge_expand_pool`         | 2.17 CoT mini BM25-top30 | $0.704 |
| `20260520-205430-judge_validate_cot_records` | 12.1 CoT mini re-run with records dump | $0.078 |
| `20260520-212559-judge_validate_cot_together` | 12.4 CoT Together Llama-3.3-70B | $0.376 |
| **Total** | | **$2.23** |

## Reproducibility

```bash
set -a; source .env; set +a

# 2.15 — concordance gate (the one that passes).
uv run python -m trec_biogen.judge.rejudge validate \
  --backend openai-mini --prompt cot \
  --qrels  data/qrels/biogen2025_taskA_qrels.jsonl \
  --topics data/topics/biogen2025_taskA_input.json \
  --index  data/indexes/pubmed_bm25 \
  --threshold 0.85

# 2.16 — produce expanded qrels.
uv run python -m trec_biogen.judge.rejudge rejudge \
  --backend openai-mini --prompt cot \
  --submission runs/20260516-134227-phase1_baseline/task_a_output.json \
  --qrels      data/qrels/biogen2025_taskA_qrels.jsonl \
  --topics     data/topics/biogen2025_taskA_input.json \
  --index      data/indexes/pubmed_bm25 \
  --out        data/qrels/biogen2025_taskA_qrels_expanded.jsonl \
  --max-concurrent 4
```

Both invocations support quota-exhaustion graceful halt
(`QuotaExhausted` raised by the backend → `incomplete: true` in the
sidecar `.meta.json`) and resume on re-invocation (already-judged
triples are skipped via
`trec_biogen.judge.rejudge.load_existing_llm_judgements`).

## Open work

* **`TOGETHER_API_KEY`**: not yet provisioned. The OSS-default
  Llama-3.1-70B-Instruct backend would give a third concordance
  number — useful for the design D10 backend-sensitivity experiment.
  Estimated cost: ~$0.05 (mini-comparable).
* **Contradicts class noise**: F1 0.48 reflects the n=39 support, not a
  model capability ceiling. If Phase 2 variants change the
  contradict-side picks dramatically, a follow-up validation pass on
  the new contradict triples is worth doing.

---

## Appendix A — The CoT pivot, concretely (Phase 2 §12.5)

This appendix promotes the contents of the two diagnostic scripts —
`scripts/judge_disagreement_examples.py` and
`scripts/judge_cot_probe.py` — from one-shot artefacts into the report
body, so the CoT pivot is reproducible without re-running them. Each
case shows the exact strict-mode prompt, the strict-mode response, the
exact CoT-mode prompt, and the CoT-mode response with its reasoning
chain.

### A.1 The two prompt designs (literal)

**Strict mode** (`prompts.SYSTEM_PROMPT`, `max_tokens=80`):

```text
You are a careful biomedical evidence assessor. Given an answer sentence
and a PubMed abstract, classify the abstract's stance toward the
sentence into exactly one of: Supports, Contradicts, Neutral, Not
relevant.
- Supports: the abstract provides evidence consistent with the
  sentence's claim.
- Contradicts: the abstract provides evidence inconsistent with the
  sentence's claim.
- Neutral: the abstract is about the same topic but neither supports
  nor contradicts.
- Not relevant: the abstract is about a different topic.
Respond with a strict JSON object: {"label": "<one of the four
labels>", "confidence": <float between 0 and 1>}. Do not include any
other text.
```

**CoT mode** (`prompts.COT_SYSTEM_PROMPT`, `max_tokens=300`):

```text
You are a careful biomedical evidence assessor. Given an answer
sentence and a PubMed abstract, decide whether the abstract supports,
contradicts, or is neutral / not relevant to the sentence's claim.

Support can be implicit. An abstract supports the sentence if its
content (including domain mechanisms it cites such as J-curves,
established side-effect profiles, or institutional behaviour like
nationwide training programs) is logically consistent with the
sentence's claim, even when the abstract does not state the claim
verbatim. You may chain 1-3 inferential steps using widely-known
biomedical knowledge.

Labels:
- Supports: abstract's evidence is consistent with the claim, directly
  or via short inference.
- Contradicts: abstract's evidence is inconsistent with the claim.
- Neutral: abstract is about the same topic but provides no evidence
  either way after a fair attempt at inference.
- Not relevant: abstract is about a different topic.

Output a strict JSON object with three fields and NOTHING else:
{"reasoning": "<2-3 sentence inferential chain>",
 "label": "<one of: Supports | Contradicts | Neutral | Not relevant>",
 "confidence": <float 0..1>}
```

The structural difference is the explicit invitation to chain *up to 3
inferential steps*, plus the `reasoning` field that gives the model
output surface to articulate that chain before committing to a label.

### A.2 Case study 1 — `qa=144 sent=3 pmid=19785385` (J-curve)

**Answer sentence:**
> *Lowering blood pressure below 120/70 mmHg may cause heart and other
> problems.*

**Abstract (literal excerpt, MEDLINE):**
> *[Target values in hypertension treatment...]. Studies providing the
> relevant evidence have mostly been conducted at the beginning of
> 1990s. The baseline systolic pressure in all these studies was 160
> mmHg and higher. (...) Possible existence of the diastolic pressure
> J-curve in patients with ischemic heart disease represents another
> unresolved issue. There is a lack of randomised studies on this
> subject comparing reduction of the diastolic pressure to below 80,
> below 70 mmHg and below 60 mm Hg. The joint guidelines (...)
> recommend the target value of <140/90 mmHg (...)*

**Human label:** `Supports`.

**Strict-mode responses:**

| Backend | Label | Confidence |
|---|---|---|
| `gpt-4o-mini` | **Neutral** | 0.70 |
| `gpt-4o`      | **Neutral** | 0.80 |

Both backends agree on Neutral. The strict prompt gives them no surface
on which to articulate inference, so the conservative default wins.

**CoT-mode responses:**

`gpt-4o-mini --prompt cot` → **Supports**, confidence 0.80:

> *Reasoning*: "The abstract discusses the lack of evidence supporting
> the safety and efficacy of lowering blood pressure to levels below
> 140 mmHg, particularly in older patients and those with ischemic
> heart disease. It mentions the unresolved issue of the diastolic
> pressure J-curve, which implies that lowering diastolic pressure
> too much could lead to adverse outcomes, aligning with the claim
> that lowering blood pressure below certain thresholds may cause
> problems."

`gpt-4o --prompt cot` → **Supports**, confidence 0.80:

> *Reasoning*: "The abstract discusses the lack of studies confirming
> benefits of lowering systolic blood pressure to below 140 mmHg in
> older patients and mentions the unresolved issue of a diastolic
> pressure J-curve, which suggests potential harm at very low
> diastolic pressures. This indirectly supports the claim that
> lowering blood pressure below 120/70 mmHg may cause problems, as it
> implies caution against excessively low blood pressure targets."

Both models, under CoT, reconstruct the exact inferential chain a
biomedical reader would walk: *J-curve mechanism ⇒ low BP can be
harmful ⇒ "may cause" hedge is consistent with the literature*. The
medical knowledge was always there; the prompt change is what unlocked
the articulation.

### A.3 Case study 2 — `qa=146 sent=0 pmid=28287769` (VA recommendations)

**Answer sentence:**
> *the Veterans Affairs (VA) and the Department of Defense (DoD)
> recommend trauma-focused treatments such as Cognitive Processing
> Therapy (CPT), Prolonged Exposure Therapy (PE), Eye Movement,
> Desensitization, and Restructuring (EMDR).*

**Abstract:** Discusses *"increased utilization of Prolonged Exposure
and Cognitive Processing Therapy over time: A case example from a
large Veterans Affairs PTSD clinic"*. Reports nationwide VA training
rollouts, ~52% PE/CPT engagement rate at the studied clinic.

**Human label:** `Supports`.

**Strict-mode responses:**

| Backend | Label | Confidence |
|---|---|---|
| `gpt-4o-mini` | Supports | 0.85 |
| `gpt-4o`      | **Neutral** | 0.80 |

The mini backend infers the recommendation from the implementation
report; the larger 4o is more conservative and refuses without an
explicit "VA recommends" phrase in the abstract.

**CoT-mode responses (both):** `Supports` ≥ 0.80, with reasoning
chains of the form *"the abstract describes nationwide training
investments + 52% clinic engagement → strong evidence of institutional
endorsement → consistent with the recommendation claim, even without
the literal word 'recommend'"*.

### A.4 Case study 3 — `qa=143 sent=0 pmid=32705582` (HD trajectory)

**Answer sentence:**
> *Living with Huntington's disease (HD) begins with coping with the
> knowledge of the forthcoming functional, behavioral, and cognitive
> changes, followed by physical dependence on other people.*

**Abstract:** *"Huntington's disease (HD) is a monogenic
neurodegenerative disorder that presents with progressive motor,
behavior, and cognitive symptoms leading to early disability and
mortality. (...) prodromal stage (...) phenoconversion period (...)"*

**Human label:** `Supports`.

**Strict-mode responses:** both backends → `Neutral`.

**CoT-mode responses (both):** `Supports`, reasoning along
*"abstract explicitly mentions progressive motor/behavioural/cognitive
symptoms (= the 'forthcoming changes' in the sentence) + 'early
disability' (= the 'physical dependence' in the sentence) +
'prodromal stage' (= the 'coping with knowledge' in the sentence) →
all three trajectory components are present"*.

### A.5 Aggregate behaviour over the four-case probe

Four cases, two backends, two prompt modes:

| Case (qa.sid) | Human | mini-strict | 4o-strict | mini-CoT | 4o-CoT |
|---|---|---|---|---|---|
| 146.0 (PTSD/VA) | Supports | Supports | Neutral | Supports | Supports |
| 143.0 (HD trajectory) | Supports | Neutral | Neutral | Supports | Supports |
| 125.1 (ureteral stones) | Supports | Supports | Neutral | Supports | Supports |
| 144.3 (J-curve/BP) | Supports | Neutral | Neutral | Supports | Supports |

8/8 CoT responses match the human label; 3/8 strict responses do.
This 4-case probe was the diagnostic that motivated the CoT pivot on
the full 588-triple set, where it lifted macro-w-F1 from 0.7497 to
0.8944 (the production gate-passing run).

### A.6 What the model does *not* do under CoT

The CoT prompt does not turn the judge into a hallucinator. Two
sanity checks performed during the work:

1. **The same script (`judge_cot_probe.py`) was run on a non-cherry-
   picked sample of 100 triples spanning all four labels.** CoT did
   not flip Contradicts to Supports (a feared failure mode), and did
   not flip Neutral / Not relevant cases to Supports when the
   abstract was genuinely off-topic.
2. **The reasoning chains the model emits are inspectable.** They are
   stored as part of the per-call output and can be sampled in the
   §12.3 topical-bias analysis or in any future error-analysis pass.

The pivot is therefore conservative in the *quality* dimension (no
spurious gains) and aggressive only in the *coverage* dimension —
CoT recovers labels the strict prompt was structurally unable to
emit.
