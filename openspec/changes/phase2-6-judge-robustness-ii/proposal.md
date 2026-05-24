## Why

Phase 2.5 closed the two-judge robustness question on the §2.17 expand-pool (`mini-cot ∩ Llama-3.3-70B-cot`) and surfaced two methodological frontiers that the relatório explicitly leaves open: (a) the §10.2 isotonic-PAV calibration of judge confidences is fit and evaluated on the same 588 gold triples, so the reported ECE numbers (~0.003 mini, ~0.000 Together) are in-sample upper bounds rather than held-out estimates; and (b) the two-judge intersection-on-contradicts pool, while conservative, still rests on only two judges from the OpenAI / Meta-Llama families — a third independent biomedical-domain backend would let us report Krippendorff α instead of pairwise Jaccard and derive a stricter three-way intersection pool. Closing both is cheap (~$3 API spend, ~1 afternoon wall-clock) and converts the calibration / cross-judge claims from "best-effort under noise" to "defensible to peer review".

## What Changes

- Add a third LLM-judge backend `qwen` (Qwen2.5-72B-Instruct served via HF Inference Providers; **pivot from Mixtral** — at implementation time HF Providers had removed all Mistral models from the chat-routable roster and Together-direct hit a 402, so the equivalent-intent substitute is Qwen2.5-72B routed by HF to OpenRouter) to the existing two-backend roster. Validate against the 588-triple human gold set under the 0.85 macro-w-F1 gate; abort the rest of the change if the gate fails.
- Rejudge the same 5 398 §2.17 candidate triples with `qwen --prompt cot` and persist the result alongside the existing `_expanded_*.jsonl` files. Hard cost cap: $5 across gold + expand-pool.
- Generalise the intersection-pool helper to **N ≥ 2 inputs** so the Phase 2.5 two-judge pool and the new three-judge pool flow through the same code path. Emit `biogen2025_taskA_qrels_intersection_3way.jsonl`.
- Add `--qrels-pool=intersection-3way` to the metrics CLI; re-score every run dir on the new pool; regenerate `reports/phase2_summary.md` with the new column.
- Add a `krippendorff_alpha(records_a, records_b, records_c, *, classes)` helper for nominal-data α; report the three-way α alongside the existing pairwise Jaccard in the judge-intersection report.
- Replace the in-sample ECE in `reports/llm_judge_calibration.md` with a **k-fold cross-validated ECE** (k=5, folds split by `qa_id` to avoid topical leakage) for both existing CoT backends. Keep the raw uncalibrated ECE unchanged (the in-sample caveat only bites on the post-isotonic numbers).
- Update `reports/judge_intersection_analysis.md` with the three-way intersection table (cell-level bootstrap CIs) and the Krippendorff α; add §10.9 to `docs/phase2_report.md` (~400–600 words) summarising the result.

## Capabilities

### New Capabilities
<!-- none — all changes extend existing capabilities -->

### Modified Capabilities
- `llm-judge`: ADD `qwen` backend (Qwen2.5-72B-Instruct via HF Providers) in `BACKEND_REGISTRY`; ADD k-fold cross-validated ECE method to the calibration spec (replacing in-sample isotonic ECE as the reported number); GENERALISE intersection-pool emission to N≥2 inputs.
- `evaluation`: ADD `--qrels-pool=intersection-3way` flag with the same source-attribution semantics as the existing intersection pool; ADD `krippendorff_alpha(records_a, records_b, records_c, *, classes)` helper exposed by `eval/metrics.py`.

## Impact

- **Code**: `src/trec_biogen/judge/backends.py` (new `HFQwen72B` adapter), `src/trec_biogen/judge/intersection.py` (N-way generalisation), `src/trec_biogen/eval/calibration.py` (new k-fold CV function), `src/trec_biogen/eval/metrics.py` (new pool enum + Krippendorff helper), `src/trec_biogen/eval/phase2_summary.py` (new column).
- **Data artefacts**: `data/qrels/biogen2025_taskA_qrels_expanded_qwen.jsonl` (new); `data/qrels/biogen2025_taskA_qrels_intersection_3way.jsonl` (new); existing `biogen2025_taskA_qrels_intersection.jsonl` untouched.
- **Reports**: `reports/llm_judge_calibration.md` (replace in-sample with held-out ECE), `reports/judge_intersection_analysis.md` (add 3-way section + α), `reports/phase2_summary.md` (auto-regenerated with new column), `docs/phase2_report.md` §10.9 (new subsection).
- **Tests**: extend `tests/test_intersection_pool.py` (N≥2 path), extend `tests/test_calibration.py` (k-fold ECE), new `tests/test_krippendorff.py` (worked-example fixture from Hayes & Krippendorff 2007), extend `tests/test_judge_rejudge_multibackend.py` (Qwen smoke test using `RecordedBackend`).
- **Dependencies**: no new Python dependencies (HF Inference Providers route already wired for Phase 2.5; Krippendorff α implemented inline against the nominal-data formula — no `krippendorff` PyPI dep).
- **API spend**: hard cap $5; estimated $2–3 (Qwen2.5-72B on OpenRouter via HF Providers ≈ $0.90/M tokens combined at current published rates; ~3× the original Mixtral budget but still inside cap).
- **Reproducibility anchor**: the §6.5 anchor (`--qrels-pool=expanded --source=human` recovers 44.34 byte-for-byte) must continue to pass; extend the existing test to also cover the `intersection-3way` path.
