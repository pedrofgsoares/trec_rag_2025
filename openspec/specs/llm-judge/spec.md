## Purpose

LLM-as-judge subsystem for expanding the human qrels with novel-PMID classifications, with a multi-backend abstraction, a 0.85 macro-w-F1 concordance gate against the human gold set, and N-judge intersection-pool emission.
## Requirements
### Requirement: LLM judge classifies (sentence, abstract) pairs into four classes
The system SHALL provide a `Judge` module that, given an `(answer_sentence, pmid, abstract_text)` triple, returns exactly one label from `{Supports, Contradicts, Neutral, Not relevant}` together with a per-call confidence score in `[0, 1]` and the input/output token counts that produced the verdict.

#### Scenario: Successful classification
- **WHEN** `Judge.classify("Aspirin reduces stroke risk.", "12345678", "<abstract text>")` is called
- **THEN** the return value is a structured record `{label: "Supports"|"Contradicts"|"Neutral"|"Not relevant", confidence: float, input_tokens: int, output_tokens: int, backend: str}`

#### Scenario: Empty or missing abstract
- **WHEN** the `abstract_text` argument is empty or `None`
- **THEN** the judge SHALL NOT call the backend; it returns `{label: "Not relevant", confidence: 1.0, ...}` deterministically and records a `skip_reason: "empty_abstract"` in the structured output

### Requirement: Backend abstraction with three concrete implementations
The judge SHALL implement a `Backend` interface with at minimum three concrete implementations: `TogetherLlama70B` (default, uses `meta-llama/Llama-3.1-70B-Instruct-Turbo` via Together.ai), `OpenAIMini` (`gpt-4o-mini`), and `OpenAI4o` (`gpt-4o`). Selecting a backend SHALL be a single Hydra override (`judge.backend=together|openai-mini|openai`).

#### Scenario: Backend selection via Hydra
- **WHEN** the operator runs `python -m trec_biogen.judge.rejudge +judge.backend=openai-mini`
- **THEN** the rejudge run uses the `OpenAIMini` backend and `metadata.yaml` records `backend: "openai-mini"` and the model identifier

#### Scenario: Missing credentials for chosen backend
- **WHEN** the chosen backend's required environment variable (`TOGETHER_API_KEY`, `OPENAI_API_KEY`) is unset
- **THEN** the rejudge CLI exits non-zero before any LLM call with an error naming the missing variable

### Requirement: Concordance gate against human-labeled pool
Before any LLM-judge run emits expanded qrels, the judge SHALL be validated against the existing 588 human-labeled triples in `data/qrels/biogen2025_taskA_qrels.jsonl`. The judge SHALL be classified as PASSING only if the per-class weighted F1 against the human labels is ≥ 0.85. A failing judge SHALL block expanded-qrels emission and the failure SHALL be reported per-class so the operator can diagnose which label suffers most.

#### Scenario: Judge passes the concordance gate
- **WHEN** the validation routine compares the LLM's labels for the 588 triples against the human labels and the macro weighted F1 ≥ 0.85
- **THEN** validation prints "OK" with the per-class F1 breakdown and writes `reports/llm_judge_validation.md`; the rejudge CLI proceeds to novel-PMID classification

#### Scenario: Judge fails the concordance gate
- **WHEN** the macro weighted F1 < 0.85
- **THEN** the rejudge CLI exits non-zero with a per-class F1 breakdown, writes the validation report including a confusion matrix, and does NOT emit `biogen2025_taskA_qrels_expanded.jsonl`

### Requirement: Expanded qrels emission preserves the official schema
The rejudge CLI SHALL emit `data/qrels/biogen2025_taskA_qrels_expanded.jsonl` in the exact same JSONL schema as `data/qrels/biogen2025_taskA_qrels.jsonl` (`qa_id`, `sentence_id`, `pmid`, `class`, `relevance`). Records SHALL include two additional optional fields: `source` (one of `"human"`, `"llm-together-llama-3.1-70b"`, `"llm-openai-mini"`, `"llm-openai-4o"`) and `confidence` (float in `[0,1]`). Human records SHALL be copied unchanged from the input qrels.

#### Scenario: Existing eval module reads expanded qrels unchanged
- **WHEN** `python -m trec_biogen.eval.metrics --qrels data/qrels/biogen2025_taskA_qrels_expanded.jsonl` is invoked
- **THEN** the existing parser reads it correctly and produces a metrics report; no code change in `eval/qrels.py` is required

#### Scenario: Source attribution is preserved per record
- **WHEN** any record in the expanded qrels file is loaded
- **THEN** its `source` field identifies whether the label came from a human assessor or which LLM backend produced it

### Requirement: Cost accounting per rejudge run
Every rejudge invocation SHALL record total cost incurred. `metadata.yaml` in the rejudge run directory SHALL include `judge_cost_usd: float`, `judge_token_breakdown: {input_tokens, output_tokens, cache_hit_rate}`, and `judge_backend: str`. A `--cost-cap=$X` CLI flag SHALL abort the run before exceeding the cap with a clean message naming the cap and the spend at abort time.

#### Scenario: Cost cap is reached mid-run
- **WHEN** a rejudge run is invoked with `--cost-cap=5` and the running spend reaches $5
- **THEN** the run halts after the in-flight batch completes, writes a partial expanded-qrels file with a clear `incomplete: true` flag in its own metadata, and exits non-zero

#### Scenario: Cost is recorded for the operator
- **WHEN** a rejudge run completes successfully
- **THEN** the operator can read `judge_cost_usd` and `judge_token_breakdown` from `metadata.yaml` without invoking any reporting tool

### Requirement: Backend-comparison sanity check
The judge module SHALL provide a `compare-backends` subcommand that re-classifies a fixed 200-pair sample with two or more named backends and reports pairwise per-class agreement plus a confusion matrix between the backends.

#### Scenario: Comparing two backends on the validation sample
- **WHEN** the operator runs `python -m trec_biogen.judge.compare_backends --backends together,openai-mini --sample 200`
- **THEN** a single report is written to `reports/llm_judge_backend_comparison.md` showing per-class agreement, total cost per backend, and a 4×4 confusion matrix per backend pair

### Requirement: Multi-judge rejudge over a fixed candidate set
The judge module SHALL support running the rejudge CLI more than once over the same candidate-triple set (same `--submission`, same `--mode expand-pool`) with different `--backend` values, each emitting to its own qrels file under `data/qrels/biogen2025_taskA_qrels_expanded_<backend-tag>.jsonl` (e.g. `expanded.jsonl` for the canonical mini-cot default; `expanded_together.jsonl` for Together-Llama-3.3-70B). Each file SHALL preserve the same JSONL schema as the canonical expanded qrels and carry `source: "llm-<backend-tag>"` per LLM record.

#### Scenario: Second-backend rejudge does not mutate the first
- **WHEN** the operator runs `judge.rejudge expand-pool --backend together --out data/qrels/biogen2025_taskA_qrels_expanded_together.jsonl` against the same candidate set as a prior mini-cot rejudge
- **THEN** the canonical `expanded.jsonl` is unchanged, the new `expanded_together.jsonl` is written, and both can be loaded independently by the same parser

#### Scenario: Backend tag inferred from --backend
- **WHEN** the `--out` flag is omitted on a multi-judge rejudge invocation
- **THEN** the output filename is derived as `expanded_<backend-tag>.jsonl` using the backend's canonical short name (`together`, `openai-mini`, `openai-4o`) so the per-backend artefact does not need explicit naming

### Requirement: Two-judge intersection pool emitter for the Contradicts class
The judge module SHALL provide an `intersection.emit_intersection_pool(records_paths, *, human_qrels, supports_source_index=0, out_path)` helper that, given a list of N ≥ 2 backend-tagged expanded qrels files (each a superset of the human qrels), emits a single qrels file containing: (a) every human-labelled record copied verbatim; (b) every Supports record from `records_paths[supports_source_index]` (default index 0, i.e. the canonical mini-cot pool) — Supports are not intersected because §12.4 shows them robust to judge choice; (c) every Contradicts record whose `(qa_id, sentence_id, pmid)` triple is present with `class=Contradicts` in **all N** input files. The emitted file SHALL be valid against the canonical qrels parser and SHALL carry a sidecar `<out_path>.meta.json` with: SHA256 of each input file, the intersection rule applied per class, the per-class positive counts before and after intersection (full and pairwise), the percentage of contradicts dropped relative to the union, and `incomplete: true` if any input was marked incomplete. The 2-input call site established in Phase 2.5 SHALL remain semantically unchanged (passing `records_paths=[a, b]` reproduces the existing two-judge pool byte-for-byte).

#### Scenario: Contradicts intersected across all N, Supports passed through
- **WHEN** N backends' expanded qrels are passed to the emitter (`records_paths=[mini, llama, qwen]`)
- **THEN** the output's Contradicts positives are the set intersection by `(qa_id, sentence_id, pmid)` across all N files, the output's Supports positives are exactly those of `records_paths[supports_source_index]`, and the human positives are bitwise-identical to the input human qrels

#### Scenario: Two-input call reproduces Phase 2.5 pool byte-for-byte
- **WHEN** `emit_intersection_pool(records_paths=[mini_path, llama_path], human_qrels=h, out_path=p)` is called with the exact same inputs that produced the canonical Phase 2.5 two-judge pool
- **THEN** the output file at `p` is byte-for-byte identical to the archived `data/qrels/biogen2025_taskA_qrels_intersection.jsonl`

#### Scenario: Sidecar metadata covers pairwise and full intersection
- **WHEN** the emitter completes for N ≥ 3 inputs
- **THEN** `<out_path>.meta.json` includes per-class positive counts after the full N-way intersection AND after every pairwise intersection (for diagnostic comparison against the Phase 2.5 two-judge pool)

#### Scenario: Any input incomplete propagates
- **WHEN** any of the N input qrels' sidecar metadata carries `incomplete: true`
- **THEN** the emitted intersection pool's sidecar also carries `incomplete: true` and names the offending input(s)

#### Scenario: supports_source_index out of range
- **WHEN** `supports_source_index` is ≥ `len(records_paths)` or negative
- **THEN** the function raises `ValueError` before reading any input file

### Requirement: Qwen2.5-72B-Instruct backend via HF Inference Providers
The judge module SHALL provide an `HFQwen72B` backend (registered as `--backend qwen`) that submits `(sentence, abstract, pmid)` triples to `Qwen/Qwen2.5-72B-Instruct` through the HuggingFace Inference Providers router (which auto-routes to OpenRouter) using the same OpenAI-compatible Chat Completions interface that the Llama route uses. The backend SHALL read `HF_TOKEN` from the environment, SHALL record `name: "hf-qwen-2.5-72b"` in `metadata.yaml`, and SHALL participate in the existing concordance-gate, cost-cap, and incremental-checkpoint machinery without modification. *(Pivot note: design D1 originally specified `Mixtral-8x7B-Instruct-v0.1` via the same route; HF Providers removed the Mistral family from the chat-routable roster in 2026-Q2 and Together-direct returned 402 on the same model, so Qwen2.5-72B is the equivalent-intent substitute that is actually routable.)*

#### Scenario: Qwen backend selection
- **WHEN** the operator runs `python -m trec_biogen.judge.rejudge validate --backend qwen --prompt cot --qrels data/qrels/biogen2025_taskA_qrels.jsonl --topics data/topics/biogen2025_taskA_input.json --index data/indexes/pubmed_bm25`
- **THEN** the gold-set 588-triple validation runs through Qwen2.5-72B and writes `reports/llm_judge_validation_qwen.md` plus per-call records to the standard `--records-out` path

#### Scenario: Qwen fails the 0.85 concordance gate
- **WHEN** the macro weighted F1 of Qwen on the 588-triple gold set is < 0.85
- **THEN** the rejudge CLI exits non-zero, writes the failure report with per-class F1 and confusion matrix, and does NOT emit `data/qrels/biogen2025_taskA_qrels_expanded_qwen.jsonl`

#### Scenario: Missing HF token
- **WHEN** `--backend qwen` is selected and `HF_TOKEN` is unset
- **THEN** the rejudge CLI exits non-zero before any LLM call with an error naming `HF_TOKEN` (mirroring the existing `--backend hf-llama` behaviour)

### Requirement: k-fold cross-validated ECE for calibration reporting
The judge module SHALL provide a `calibration.kfold_ece(records, *, k=5, n_bins=10, seed=0)` function that splits the input per-call records into `k` topic-disjoint folds (folds assigned by `qa_id` to prevent topical leakage), fits the isotonic-PAV calibration mapping on `k-1` folds, predicts on the held-out fold, aggregates predictions across folds, and returns both the raw and post-calibration ECE as `{ece_raw_mean, ece_raw_std, ece_calibrated_mean, ece_calibrated_std, n_per_fold}` over the `k` folds. The reported calibration ECE in `reports/llm_judge_calibration.md` SHALL be the held-out mean from this function, not the in-sample isotonic fit of Phase 2.

#### Scenario: Folds are topic-disjoint
- **WHEN** `kfold_ece(records, k=5)` is called on records spanning 40 distinct `qa_id` values
- **THEN** every `qa_id` appears in exactly one held-out fold and the same `qa_id` never appears in both the training and test partition of any single fold

#### Scenario: Held-out ECE replaces in-sample ECE in the calibration report
- **WHEN** `reports/llm_judge_calibration.md` is regenerated for either CoT backend (`openai-gpt-4o-mini` or `together-llama-3.3-70b`)
- **THEN** the post-calibration ECE column reports the k=5 held-out mean ± std (not the in-sample fit), and the report explicitly cites `qa_id`-disjoint folds as the splitting rule

#### Scenario: Raw uncalibrated ECE is unchanged
- **WHEN** the calibration report is regenerated
- **THEN** the raw (uncalibrated) ECE column for each backend matches the Phase 2 number byte-for-byte (the in-sample caveat only bites on the post-isotonic estimate; raw ECE is fit-free)

