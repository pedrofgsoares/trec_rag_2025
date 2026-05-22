# LLM-Judge Topical Bias Analysis — Phase 2 §12.3

Per-topic distribution of the 4170 LLM-emitted positives in `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`. Diagnostic for whether the judge systematically overgenerates supports or concentrates positives in particular topical clusters.

## Aggregate

- Topics with ≥ 1 LLM positive: **40**
- LLM support positives total: **3807**
- LLM contradict positives total: **363**
- Global support/contradict ratio: **10.49**

Per-topic LLM-support count:
- mean: 95.2, median: 88, IQR: [62, 116]

## Top-10 topics by LLM-support density

Topics where the LLM judge accepted the most novel-support claims. High counts here may indicate either (a) a productive topical area with plentiful supporting evidence in PubMed, or (b) judge overgeneration — manual spot-check the top-3 before trusting.

| qa_id | question (truncated) | LLM sup | LLM con | sup/con | human sup | mean conf |
|---|---|---:|---:|---:|---:|---:|
| 134 | what are the symptoms and treatments for ibs? | 211 | 13 | 15.1 | 36 | 0.86 |
| 126 | what is nondisjunction and what causes it? | 192 | 25 | 7.4 | 22 | 0.85 |
| 133 | What are the short and long term effects of surgery and its  | 194 | 18 | 10.2 | 30 | 0.83 |
| 127 | what effects does gene therapy have on an organism? | 198 | 7 | 24.8 | 29 | 0.85 |
| 146 | how can ptsd be treated/managed? | 140 | 10 | 12.7 | 22 | 0.86 |
| 119 | what drug or combination of drugs is most popular for treati | 138 | 7 | 17.2 | 20 | 0.86 |
| 144 | when should i treat high blood pressure? | 131 | 13 | 9.4 | 17 | 0.85 |
| 154 | how to breathe when you are short of breath? | 126 | 13 | 9.0 | 22 | 0.81 |
| 147 | what treatments options are available for atherosclerosis? | 132 | 5 | 22.0 | 12 | 0.83 |
| 117 | What will mutation in runx2 affect in the future? | 128 | 1 | 64.0 | 5 | 0.86 |

## Bottom-10 topics by LLM-support density

Topics where the LLM judge accepted few novel supports. Either (a) genuinely narrow topical evidence in PubMed, or (b) the BM25 first-stage missed evidence for this topic and the LLM had nothing to accept.

| qa_id | question (truncated) | LLM sup | LLM con | sup/con | human sup | mean conf |
|---|---|---:|---:|---:|---:|---:|
| 148 | why do i bruise so easily? | 51 | 18 | 2.7 | 15 | 0.86 |
| 128 | how does cbd effect liver enzymes? | 54 | 14 | 3.6 | 6 | 0.88 |
| 140 | what can cause acute fractures in both of your hands when th | 49 | 10 | 4.5 | 6 | 0.84 |
| 118 | how long does a minor cornea injury take to heal without med | 44 | 11 | 3.7 | 3 | 0.81 |
| 120 | what causes left sided facial numbness? | 53 | 1 | 26.5 | 6 | 0.85 |
| 151 | what has turned my skin gray and scaly? | 54 | 0 | 54.0 | 10 | 0.81 |
| 153 | if hct is 34, does pt need blood transfusion? | 38 | 12 | 2.9 | 6 | 0.83 |
| 138 | can i use Sudafed after using Afrin for nasal congestion? | 27 | 14 | 1.8 | 11 | 0.82 |
| 150 | what are the effectiveness of physical therapy interventions | 36 | 3 | 9.0 | 0 | 0.82 |
| 155 | for how long after cataract surgery do i keep dropping eye d | 33 | 3 | 8.2 | 5 | 0.77 |

## Overgeneration check — extreme sup/con ratios

Topics where the LLM judge accepts many supports but ~zero contradicts. Cross-check whether these topics have biologically-plausible *contradiction* candidates in PubMed; if yes, the judge is biased toward `Supports` on this topical subset.

| qa_id | question | LLM sup | LLM con |
|---|---|---:|---:|
| 117 | What will mutation in runx2 affect in the future? | 128 | 1 |
| 141 | what are the possible complications for a person that has yellow fever | 105 | 0 |
| 130 | is pcos linked to oxidative stress? | 104 | 0 |
| 151 | what has turned my skin gray and scaly? | 54 | 0 |
| 120 | what causes left sided facial numbness? | 53 | 1 |

## Per-class confidence distribution (LLM rows only)

| Class | n | mean conf | median | min | low-confidence (<0.7) |
|---|---:|---:|---:|---:|---:|
| Supports | 3807 | 0.840 | 0.850 | 0.700 | 0 (0.0%) |
| Contradicts | 363 | 0.864 | 0.900 | 0.800 | 0 (0.0%) |

## Interpretation

The global support/contradict ratio is **10.5** — consistent with PubMed's known prior toward affirmative findings. Treat with mild caution but not a strong bias signal.

Data sources:
- expanded qrels: `data/qrels/biogen2025_taskA_qrels_expanded.jsonl`
- topic questions: `data/topics/biogen2025_taskA_input.json`
