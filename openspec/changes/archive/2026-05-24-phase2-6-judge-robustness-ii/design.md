## Context

Phase 2.5 (archived 2026-05-22, tag `phase2.5-baseline`) introduced the two-judge intersection-on-contradicts pool (`mini-cot ∩ Llama-3.3-70B-cot`) and tightened the §10.2 calibration narrative with a post-review caveat: the isotonic-PAV ECE was *fit and evaluated on the same 588 gold triples*. The relatório now carries that caveat explicitly in §10.2 and forward-references this change for closure. Two artefacts shape this work:

- [`src/trec_biogen/judge/backends.py`](../../../src/trec_biogen/judge/backends.py) already factors backends through `HTTPBackend` + `BACKEND_REGISTRY`; the HF Inference Providers adapter (`hf-llama`) added in Phase 2.5 is the closest template for the new Qwen entry (originally Mixtral — see D1 pivot).
- [`src/trec_biogen/judge/intersection.py`](../../../src/trec_biogen/judge/intersection.py) currently hardcodes a two-input signature (`records_a`, `records_b`); this change generalises it to N inputs without changing the existing call site's semantics.

The Phase 2.5 cross-judge agreement on contradicts was *strongly asymmetric* (Jaccard 0.93 Supports, 0.12 Contradicts). A third judge from a different model family (Qwen2.5-72B, Alibaba dense — see D1 pivot from the originally planned Mixtral-8x7B) gives us a coordinate to disentangle "model family effect" from "true label ambiguity" on the contradict class.

## Goals / Non-Goals

**Goals:**

- Replace the in-sample ECE in §10.2 with a held-out, `qa_id`-fold-disjoint estimate so the calibration claim survives peer review.
- Add a third independent LLM judge and report a three-way Krippendorff α on the 588-triple gold set. The α is the headline statistic; pairwise Jaccards are kept as a sanity check.
- Derive a three-way intersection-on-contradicts pool (`mini ∩ Llama ∩ Qwen`) and re-score every existing run dir against it.
- Keep all changes additive: the existing two-way intersection pool, expanded pool, and official pool remain reproducible byte-for-byte after this change lands.

**Non-Goals:**

- We do **not** add a fourth backend in this change. Closure of "judge robustness as a methodology" is bounded at three independent backends; further expansion is a separate change.
- We do **not** re-run any Phase 1 / Phase 2 ablation. All pipeline runs are reused from existing `runs/*/task_a_output.json`; only the scoring pool changes.
- We do **not** introduce a new dependency on `krippendorff` (PyPI). The nominal-data α formula is ~30 lines and we own the test fixture; an external dep would force a transitive transitive-dependency review for a single function.
- We do **not** ship a CLI command for k-fold ECE re-fitting beyond a single `judge calibrate-cv` subcommand. The held-out ECE is a *report* number, not an operational artefact — no other code consumes it.

## Decisions

### D1 — Qwen2.5-72B-Instruct via HF Providers (pivot from Mixtral-8x7B)

**Original decision (superseded)**: Mixtral-8x7B-Instruct via HF Inference Providers.

**Pivot (2026-05-23, at implementation time)**: Qwen2.5-72B-Instruct via HF Providers (HF routes to OpenRouter under model id `Qwen/Qwen2.5-72B-Instruct`).

**Why the pivot was forced**:

- HF Inference Providers returns `400 "not a chat model"` for every Mistral-family chat model probed: `Mixtral-8x7B-Instruct-v0.1` (with and without `:together` / `:fireworks-ai` provider pin), `Mistral-7B-Instruct-v0.3`, `Mistral-Small-24B-Instruct-2501`. HF removed the Mistral family from the chat-routable roster in 2026-Q2.
- The Together-direct fallback (which is how the design originally proposed reaching Mixtral if HF was unavailable) returned HTTP 402 `Credit limit exceeded` — Together account has run out of prepaid balance. Topping up the account is contrary to the "OSS-first, bounded paid-API exceptions" project doctrine and would also break the OSS-default story for the third judge.
- Probed alternatives on HF Providers (single-call smoke test with `model="X"`, `max_tokens=5`): Qwen2.5-72B-Instruct → 200 OK (routed to OpenRouter); DeepSeek-V3 → 200 OK (routed to DeepSeek); Qwen3-32B → 200 OK; Gemma-2-27B → 400 (no provider); Cohere Command R+ → 400 (does not exist on the registry).

**Why Qwen2.5-72B specifically**:

- *Different model family*: Alibaba lineage, distinct from both OpenAI's GPT-4 (`openai-mini`) and Meta's Llama-3 (`together`, `hf-llama`). The original "third independent family" intent is preserved.
- *Comparable scale*: 72B dense matches Llama-3.3-70B in capacity; not a regression to the small-model risk that ruled out BioMistral-7B at proposal time.
- *Cost*: ~$0.90 per M tokens combined on OpenRouter as of May 2026 (3.3× the projected Mixtral rate; still inside the $5 cap — projected $1.5–2 for gold + expand-pool combined).
- *Routing reliability*: OpenRouter is a single hop (no auto-routing across N providers), so latency and 5xx behaviour are predictable for the cost-cap and incremental-checkpoint machinery to track.

**Rejected at pivot time**:

- **DeepSeek-V3**: viable but the per-output pricing ($1.10/M output) and variable latency under MoE make cost projection harder. Reserved as the fallback if Qwen fails the gate.
- **Qwen3-32B**: cheaper but at 32B sits closer to the BioMistral-7B size cluster that the original D1 ruled out for gate-failure risk.
- Carry-over rejections (BioMistral-7B, MedPaLM-2, Claude-Haiku) still hold from the original D1 for the same reasons.

**Spec impact**: the `mixtral` backend identifier in the `llm-judge` spec is replaced by `qwen`; the registered class is `HFQwen72B` (filename and registry name updated); all data-artefact filenames change `_mixtral` → `_qwen`. No structural change to the design otherwise.

### D2 — Generalise `emit_intersection_pool` to N inputs, keep 2-input API

**Decision**: change the signature to `emit_intersection_pool(records_paths: list[Path], *, human_qrels_path, supports_source_index: int = 0, out_path)`. The 2-input call site stays valid (`records_paths=[a, b]`).

**Why**:

- The semantics generalise cleanly: Supports come from `records_paths[supports_source_index]` (mini by default); Contradicts kept only when all N paths label the same triple as Contradicts; human positives bitwise-identical pass-through.
- Keeps a single source of truth for the intersection logic — the Phase 2.5 two-way pool and the Phase 2.6 three-way pool are produced by the same function with different inputs.
- Sidecar metadata (`<out>.meta.json`) extends naturally: SHA256s for all N inputs, per-class counts before/after for each pairwise and full intersection, percentage of contradicts dropped per intersection step.
- Rejected alternative: a separate `emit_intersection_pool_3way` function. This would diverge in three weeks when someone wants 4-way or wants to re-run 2-way with new inputs.

### D3 — Krippendorff α: implement inline (nominal data, no missing values)

**Decision**: implement `krippendorff_alpha(labels_per_coder: list[list[str]], classes: tuple[str, ...]) -> float` in `eval/metrics.py` using the standard nominal-data formula:

```
α = 1 - (D_observed / D_expected)
D_observed = Σ_units Σ_pairs_in_unit δ(coder_i, coder_j) / N_pairs_per_unit
D_expected = (Σ_classes Σ_other_classes n_c * n_c' / (N_total - 1))
δ(c1, c2) = 0 if c1 == c2 else 1
```

**Why**:

- No missing values in our case (every triple is judged by every backend); the full formula with missing-value handling is unnecessary.
- ~30 lines of code; the test fixture is the worked example from Hayes & Krippendorff (2007) "Answering the call for a standard reliability measure for coding data" Table 1, which has the published α = 0.7434 — a deterministic anchor.
- Adding `krippendorff` PyPI dep brings in NumPy/Pandas pinning issues and a maintenance footprint disproportionate to one function call.
- Rejected alternatives:
  - **statsmodels.stats.inter_rater**: only ships Cohen's κ and Fleiss' κ, not α.
  - **scipy**: no α implementation as of SciPy 1.13.

### D4 — k-fold CV ECE: folds at `qa_id` boundaries, k=5, default seed=0

**Decision**: `kfold_ece(records, *, k=5, n_bins=10, seed=0) -> dict[str, float]` returns `{"ece_raw_mean", "ece_raw_std", "ece_calibrated_mean", "ece_calibrated_std"}`. Folds are constructed by hashing `qa_id` mod k to keep all triples for one topic in the same fold.

**Why folds at `qa_id`**:

- Triples in the same topic share the same biomedical sub-domain and frequently the same PMID candidates across sentences. Random per-triple folds would let the same PMID's calibration leak across train/test.
- This is the same anti-leakage discipline we use elsewhere (e.g., the §6.5 reproducibility anchor is computed per-topic).

**Why k=5**: with 40 topics the smallest fold has ~8 topics → ~117 triples on average (still above the n=30 rule of thumb for ECE stability per bin × 10 bins ≈ 300 samples, but acceptable with `n_bins=10` and pooling).

**Why seed=0**: matches the established convention in `eval/metrics.py::bootstrap_ci`.

### D5 — Cost cap enforcement: per-run `--cost-cap`, not a global ledger

**Decision**: pass `--cost-cap 2.5` to each Qwen invocation (gold-set and expand-pool). The judge backends already enforce per-run caps via existing `BACKEND_REGISTRY` machinery; relying on the per-run cap keeps the change minimal. (Post-pivot the per-call cost is ~3× the original projection; cap unchanged because the 5 398-triple expand-pool spend is still projected at ~$1.5.)

**Why not a global ledger**: a global ledger would need to read/write state across processes and survive crashes — over-engineering for two invocations whose combined budget is $5 with two-decimal-place pricing visibility.

### D6 — Third-judge gate failure → abort change, do not relax

**Decision**: if the third judge (Qwen2.5-72B per D1 pivot, originally Mixtral-8x7B) fails the 0.85 macro-w-F1 gate against the 588-triple gold set, abort the change. Document the failure in `reports/llm_judge_validation_qwen.md` and write a post-mortem; do **not** lower the gate, do **not** ship a 3-way pool with a sub-gate judge.

**Why**: the gate is the contract that makes "judge" interchangeable with "annotator" downstream. A sub-gate judge contaminates the intersection rather than reinforcing it.

**Recovery path**: if the gate fails, the proposal pivots to "post-mortem: Qwen2.5-72B does not clear the gate on biomedical entailment — implications for the broader LLM-as-judge literature". DeepSeek-V3 (rejected at D1 pivot time for output-pricing reasons) becomes the second retry candidate. The k-fold ECE work in §1 still ships (it does not depend on the third judge).

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Qwen2.5-72B fails the 0.85 gate. No published biomedical-NLI prior on Qwen2.5; the BioMistral-7B prior at ~0.78 is not directly applicable (different family, different scale) but signals the warning. | Gate check is the very first §2 task; per D6 we abort and pivot to a post-mortem rather than weakening the gate. The k-fold ECE deliverable is independent and still ships. Fallback candidate is DeepSeek-V3 (rejected at D1 pivot time for output-pricing reasons, but a viable second try). |
| HF Inference Providers routes Qwen to a provider with worse latency / higher 5xx rate than Phase 2.5's Groq-routed Llama experience. (Current routing is to OpenRouter — single hop, predictable.) | The Phase 2.5 code-review pass added incremental-checkpoint atomic writes and 5xx-retry semantics; reuse them. Run with `--max-concurrent 4` (conservative) on the expand-pool. |
| Upstream model unavailability mid-implementation (this is exactly what happened with Mixtral — D1 had to pivot to Qwen). | Smoke-test the chosen model with a single `max_tokens=5` chat-completion call *before* writing code that depends on it. Already done for Qwen → 200 OK at pivot time. |
| Three-way intersection on Contradicts collapses to near-empty (Phase 2.5 already dropped from 363 → 43 mini-Llama agreed; another intersection layer could halve again). | This *is* a finding — if α is low and the 3-way Contradicts pool has <20 positives, we report exactly that and flag the intersection as "too noisy for macro-F1 conclusions on Contradicts" in §10.9. The Supports pool is unaffected (single-source pass-through). |
| k-fold ECE with k=5 still has high variance on the small Contradicts subset (n=39 in gold). | Report ECE separately for Supports and overall, and add a `n_per_fold` column to the calibration table so the reader can judge whether the held-out estimate is stable. Do not report a Contradicts-only ECE. |
| Adding `--qrels-pool=intersection-3way` to `eval/metrics.py` and `phase2_summary.py` without breaking the §6.5 anchor. | The reproducibility test is extended (not modified) to cover the new pool. The change does not touch the default-pool code path; new enum value, new branch. |

## Migration Plan

This is an additive change; no migration needed.

- **Forward path**: the two-way intersection pool, expanded pool, and official pool remain valid and unchanged. The three-way intersection pool is a new file at a new path; existing consumers do not see it unless they pass `--qrels-pool=intersection-3way`.
- **Rollback**: revert the commit. No data migrations, no schema changes, no backwards-incompatible config flags.
- **Tagging**: if the third-judge gate passes and the three-way intersection produces a coherent Contradicts pool (≥20 positives), tag `phase2.6-baseline` on archive. Otherwise tag `phase2.6-judge-robustness-ii` (methodological hardening only — same fallback convention as Phase 2.5).

## Open Questions

- **Qwen version pin**: HF Inference Providers may upgrade `Qwen/Qwen2.5-72B-Instruct` to a 2026-Q3 release without notice. Recommendation: pin to `Qwen/Qwen2.5-72B-Instruct` explicitly; if HF deprecates, revisit in a new change (same pivot pattern we just used for Mixtral).
- **Krippendorff α reporting unit**: do we report one α for the 588-triple gold set, or one α per class (Supports α, Contradicts α)? Recommendation: report both — overall α as the headline, per-class α as a diagnostic. The cost is two extra rows in a table.
