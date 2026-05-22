## Context

The archived `phase2-pool-aware-pipeline` change closed Phase 2 with a single-judge expanded pool (`gpt-4o-mini --prompt cot` on 5398 BM25 top-30 candidate triples; 4170 LLM positives). Two §12 hardening tasks already validated the judge:

- §12.1 bootstrap CI on the concordance gate: `0.8982 [0.876, 0.920]` for mini, `0.9112 [0.886, 0.936]` for Together-Llama-3.3-70B — both backends clear the 0.85 gate at the lower CI bound.
- §12.4 multi-backend pairwise concordance on the 588 human triples: raw agreement 0.867, Cohen's κ = 0.34 ("fair"). Per-class breakdown shows Supports F1 robust across judges, Contradicts F1 carrying material variance.

The structural Phase 2 win sitting on this judge — `phase2_no_negex` beating Phase 1 on official Contradicts F1 by +2.13 pp — has therefore been validated *quantitatively* (passes gate) but not *robustly* (single judge of the two that cleared the gate). The cheapest cleanest demonstration of robustness is to re-run the §2.17 rejudge with the second backend and re-score every variant against an **intersection pool** built from the two judges' agreement on the contradicts class.

In parallel, the Phase 2 report lacks a qualitative dimension: there is no per-topic breakdown of where the pipeline gains, where it ties, and where it loses against the starter baseline. The per-cell F1 numbers already exist in each run's `metrics_2025.json`; a topic-level aggregation plus a CLI that surfaces concrete PMID-level differences across runs would let us write a 3-topic qualitative section without any new compute.

## Goals / Non-Goals

**Goals:**

- Produce a second, fully independent LLM-judge rejudge over the same 5398 candidate triples from §2.17, using `together-llama-3.3-70b --prompt cot`. Persistence is parallel to the existing mini-cot rejudge artefacts so any downstream consumer can pick either or both.
- Materialise a **two-judge intersection pool** for the Contradicts class: a triple is a positive Contradict iff *both* mini and Together independently label it Contradicts. Supports stay on the union pool (already shown robust). Human-labelled positives are passed through unchanged.
- Wire `--qrels-pool=intersection` end-to-end (load, score, summarise). The §6.5 anchor `--qrels-pool=expanded --source=human` MUST continue to reproduce 44.34 byte-for-byte.
- Surface per-topic F1 on every existing run; build a CLI that diffs two runs at the topic level and prints the PMIDs in `A \ B`, `B \ A` with their LLM-judge labels and confidences.
- Pick 3 topics — one high-margin win against starter, one tie, one loss — and produce a qualitative narrative section for `docs/phase2_report.md`. Pool source for the qualitative pass is the intersection pool.

**Non-Goals:**

- No new model training (Phase 3 NLI fine-tuning is a separate change, conditional on this one's findings).
- No new variant runs. Re-scoring is purely re-derived from cached `task_a_output.json` files in existing run dirs.
- No new prompt mode: CoT is used throughout, per the established project default.
- No human inter-annotator review (§12.9 remains out of scope).
- Not changing the published-anchor evaluation contract: the official-pool and the union-expanded-pool paths must remain bit-identical.

## Decisions

### D1 — Use Together-Llama-3.3-70B (not 3.1-70B)

The §2.15 strict-mode validation used `Llama-3.1-70B-Instruct-Turbo`; §12.4 had to switch to `Llama-3.3-70B-Instruct-Turbo` because Together moved the 3.1 endpoint to dedicated-only. 3.3 cleared the gate at `0.9112 [0.886, 0.936]`, so we treat it as the canonical Together backend for this change. Alternative considered: a third independent backend (e.g., Mistral-Large via OpenRouter). Rejected because the gate has only been demonstrated on mini-cot and Together-3.3-cot; introducing an un-gated third judge would weaken the intersection-pool argument rather than strengthen it.

### D2 — Intersection on Contradicts only, union on Supports

The §12.4 finding is asymmetric: Supports F1 carries little judge variance, Contradicts F1 carries a lot (Together tends to be more conservative on contradicts, F1 0.25 vs mini's 0.51 on the 588 human triples). Applying intersection to Supports would shrink the Supports pool unnecessarily and inflate the noise floor of every variant's Supports F1. Intersecting only Contradicts isolates the methodological intervention exactly where the variance lives. The intersection pool is therefore the union pool minus the contradicts-class triples on which the two judges disagree.

### D3 — Persist a third qrels file, not a mutation of the existing one

The expanded qrels file is the canonical artefact downstream consumers point at. Rather than mutate it in place (and risk breaking the §6.5 anchor or any prior run that pinned its contents), we emit two new sibling files:

- `data/qrels/biogen2025_taskA_qrels_expanded_together.jsonl` — Together-cot's rejudge of the 5398 triples (plus the 588 human triples copied verbatim).
- `data/qrels/biogen2025_taskA_qrels_intersection.jsonl` — derived: human positives + (mini ∩ together) for contradicts, mini-only for supports. Sidecar `.meta.json` records both source files' SHA256 + the derivation rule.

Both are gitignored under `data/`; the file SHAs and intersection rule are committed via the report.

### D4 — Per-topic aggregation as a pure post-processing pass

The per-cell F1 numbers are already produced by `eval/metrics.py` and persisted in each run's `metrics_2025.json`. Topic-level aggregation = `groupby qa_id then mean F1`. Implementing this as a post-processing helper (`eval/per_topic.py`) avoids any change to the metric contract and keeps per-cell macro as the canonical headline number.

### D5 — Qualitative analysis uses concrete PMID-level diffs, not aggregate stats

The `scripts/per_topic_diff.py` CLI takes two `(run_dir, qa_id)` arguments and prints: emitted PMIDs in each run, set difference, LLM-judge label and confidence for each PMID in `A \ B` and `B \ A` from the intersection-pool qrels. This makes the narrative grounded in specific cases — the analysis can quote PMIDs and the model's reasoning chain (`raw_response` from the rejudge records) instead of saying things like "the pipeline does better here in general".

### D6 — Topic selection is mechanical, not cherry-picked

We use the **starter_baseline run** as the comparison anchor (it's the published baseline and the most public-facing comparison). The three topics are picked by ranking `phase1_baseline.topic_f1 - starter_baseline.topic_f1` on the intersection pool and taking: the largest positive, the value nearest zero, and the largest negative. This is recorded in the qualitative report so a reader can verify the selection is not cherry-picked.

## Risks / Trade-offs

- **[Risk]** Together rejudge cost overshoots ~$1.50 budget if rate limits force serialised calls → **Mitigation**: hard `--cost-cap` flag (already implemented in `judge/rejudge.py`); resume-mode means a budget halt just produces partial intersection records (the intersection-pool emitter flags `incomplete: true` in the sidecar).
- **[Risk]** Intersection pool may shrink so much that Contradicts F1 numbers become noise-dominated → **Mitigation**: report the size of the intersection vs union pool alongside every F1, and pair the headline number with the bootstrap CI from §12.1's framework (apply the same `bootstrap_ci` helper at the cell level on the intersection-scored runs). If the intersection pool is < ~30 % of the union, flag this in the report and downweight the conclusions accordingly.
- **[Risk]** Per-topic analysis surfaces a topic where Phase 1 loses badly and the cause is opaque → **Acceptance**: this is exactly what the qualitative analysis is for. Findings reported honestly; the report value is in surfacing the loss, not in hiding it.
- **[Risk]** Together rate-limits cause the rejudge to drift past 2 h wall-clock → **Mitigation**: nohup+setsid detached launch per `feedback_long_running_llm_loops.md`; budget for half-day wall-clock not half-hour.
- **[Trade-off]** Using mini-only for the Supports pool means the intersection narrative only applies to Contradicts. We accept this asymmetry because it matches the empirical κ asymmetry — claiming intersection for Supports would over-engineer for a variance signal that does not exist there.

## Migration Plan

- No public contract changes; `--qrels-pool=intersection` is purely additive alongside `official` and `expanded`.
- Run order: (1) Together rejudge; (2) intersection-pool emitter; (3) re-score all `runs/*` dirs; (4) per-topic aggregation; (5) qualitative analysis script; (6) report writing.
- Roll-back: deleting the two new qrels files restores Phase 2 behaviour exactly. The code paths gate on file presence; absence falls back to mini-only union pool.
