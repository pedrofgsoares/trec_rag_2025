## ADDED Requirements

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
