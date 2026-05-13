# Phase-1 Gap Analysis (template — task 12.2)

> Use this template only if `phase1_pass` is False in the Phase-1 metrics
> output. If both `Supports F1 ≥ 60` AND `Contradicts F1 ≥ 10` (strict, 2025
> qrels), skip this file and proceed to 12.3.

## Run

* Run id: `<runs/<id>/>`
* Date: `<YYYY-MM-DD>`
* Git SHA: `<sha>`
* Config: `<configs/run/...>`

## Numbers (from `metrics_2025.json`)

| Setting | Class | P | R | F1 | Target | Δ |
|---|---|---:|---:|---:|---:|---:|
| strict  | support    |   |   |   | 60.00 |   |
| strict  | contradict |   |   |   | 10.00 |   |
| relaxed | support    |   |   |   |  —    |   |
| relaxed | contradict |   |   |   |  —    |   |

## Threshold(s) missed

- [ ] Supports F1
- [ ] Contradicts F1

## Proposed Phase-2 mitigation

For each missed threshold, name the next experiment and the expected lift,
referencing the documented Phase-2 candidates in
`openspec/changes/add-baseline-pipeline/proposal.md`:

* **Supports F1**: e.g. swap MedCPT-CE for a stronger biomedical reranker, or
  enable BM25+RM3 expansion in retrieval.
* **Contradicts F1**: e.g. expand the NegEx cue list (review
  `runs/<id>/negation_audit.jsonl`), or move to a fine-tuned SciFive-MedNLI
  checkpoint.

## NegEx audit summary

Pull from `runs/<id>/negation_audit.jsonl` (task 8.6). Of the 50 sampled
dropped sentences, how many were false negatives (i.e. *should* have been
kept)? If > 5/50, expand the cue list before any model change.

## Blockers / unknowns

E.g. SciFive-MedNLI checkpoint name not yet validated, official input format
deviates from documented schema, etc.
