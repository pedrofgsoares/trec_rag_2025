# Phase 2.5 §6 — Per-topic error analysis

## Methodology

Three topics were selected mechanically (no cherry-picking) by ranking
`phase1_baseline.topic_F1 − starter_baseline.topic_F1` on the intersection
qrels pool, Strict setting, Supports class, and taking:

* (a) the topic with the *largest positive* Δ (Phase 1 wins),
* (b) the topic with the *Δ closest to zero* (tied),
* (c) the topic with the *largest negative* Δ (Phase 1 loses).

The full sorted appendix is at the end of this report; the routine lives
in [`scripts/per_topic_diff.py`](../scripts/per_topic_diff.py) and the
deterministic tie-break by ascending qa_id guarantees reproducibility.

The qualitative narrative below quotes PMID-level set differences taken
from the same script's `--qa-id` mode. PMIDs labelled `[support]` /
`[contradict]` are intersection-pool positives (human + agreed-by-both-LLMs);
`[<unjudged>]` means the PMID is not in any pool source.

## Selected topics

| Pick | qa_id | Δ Sup F1 (Phase 1 − starter) | Question |
|---|---|---:|---|
| (a) Phase 1 wins | 150 | **+13.94** | what are the effectiveness of physical therapy interventions in reducing pain in patients with lumbar disc herniation? |
| (b) tied | 120 | +0.00 | what causes left sided facial numbness? |
| (c) Phase 1 loses | 131 | **-19.49** | what are side effects of using formoterol? |

## Topic 150 — Phase 1 wins (+13.94 pp)

Question is about non-pharmacological pain interventions for lumbar
disc herniation. Phase 1's MedCPT-CE-reranked picks surface PMIDs that
mini-cot rates as Supports for both sentences (`32118715`, `29200655`,
`36249665`, `39669154`). Starter's picks are mostly unjudged — only
`27942133` (sentence 1) is a mini-cot-confirmed support.

**Pattern**: Phase 1's reranker found evidence-bearing PMIDs that the
starter-kit's BM25-only path missed. The LLM judge confirmed them as
supports even though they weren't in the human pool. **This is the
pool-expansion mechanism working as designed** — the variant's intrinsic
quality is real, just invisible on the official pool.

Concrete sentence (s1): *"Specific successful procedures include
mobilization therapies, exercise, traction, and decompression."* Phase
1's PMID `36249665` was judged Supports by mini-cot with confidence 0.9;
that PMID is a 2022 RCT of decompression therapy for lumbar disc
herniation — directly on-topic, lexically less obvious than a query on
"decompression lumbar" alone, retrievable only because the MedCPT-CE
rerank could weigh both terms jointly.

## Topic 120 — tied (0.00 pp)

Question is about causes of left-sided facial numbness (trigeminal
neuropathy). Both pipelines emit 3 supports per sentence; their
intersection is empty (different picks on every cell).

**Pattern**: Phase 1's picks (sentence 0: `1521555`, `20434056`,
`38357058`; sentence 1: `20735441`, `29780224`, `38107719`) are *all
mini-cot-confirmed supports*. Starter's picks include *two human-pool
golds per sentence* (s0: `17305614`, `9329237`; s1: `30930839`,
`33907474`) plus one LLM-confirmed each. **Both reach effectively the
same Supports F1 by finding different-but-valid PMIDs** — Phase 1
trades human-pool recall for LLM-judged recall on a disjoint set.

This is pool bias visible in microcosm: on the conservative pool, two
pipelines that converge on the same *topical understanding* but pick
different *evidentiary documents* land at the same score, even though
neither covers the other's picks. The Supports F1 is high in absolute
terms (~16-17 across both runs) because the LLM judge ratifies both
sets — but the per-pipeline overlap is 0 of 6 picks. A reader interested
in actual evidence retrieval should weight this as "both pipelines find
distinct valid evidence" rather than "neither pipeline is better".

## Topic 131 — Phase 1 loses (-19.49 pp)

Question is about side effects of formoterol (asthma inhaler), with
6 sentences covering different patient populations. The loss is
**concentrated in sentences 4 and 5** (paediatric and cancer-patient
side effects); Phase 1 actually *wins* sentences 1 and 2 (general
patient safety and oral-route side effects).

* Sentence 1 (*"It is safe for most patients."*): Phase 1 picks 2
  mini-cot-confirmed supports; starter's 3 picks are all unjudged.
  Phase 1 wins this cell.
* Sentence 2 (*"tremors and palpitations…"*): Phase 1 picks `1353718`
  and `1973907` — both **human-pool golds** (confidence 0.9). Starter
  misses both.
* Sentence 4 (*"In children with asthma, the most frequent side effects
  were worsening of asthma, lung infection, cough, fever and headache."*):
  Phase 1 has 0/3 in-pool picks. Starter picks `30828592` and `37743297`,
  both human-pool golds (confidence 0.7/0.6).
* Sentence 5 (*"In cancer patients, the few side effects included
  tremors, swelling, increased heart rate, and indigestion."*): Phase 1
  has 0/2. Starter picks `1273541`, `22344584`, `7140872` — **all three
  human-pool golds** (confidence 0.7-0.9).

**Pattern**: formoterol side-effect studies have a small, well-defined
gold-standard literature concentrated on the more specific patient
populations (children, cancer patients). Those papers — small PMIDs
(`1273541`, `22344584`) and curated 2010s clinical reports
(`30828592`, `37743297`) — have strong lexical match with the specific
sub-population query terms. Phase 1's MedCPT-CE reranker, optimised for
semantic relevance to the *general* question, demotes these specific
sub-population papers in favour of broader formoterol literature that
the LLM judge had no signal to confirm (because §2.17's expand-pool only
saw Phase 1's MedCPT-CE-reranked top-30).

**This is the asymmetric cost of pool expansion**: when the human pool
*does* have good coverage on a sub-question (specific patient
populations with curated trial literature), the LLM-expanded pool's
"fairness" advantage disappears, and any pipeline that diverges from
the original assessors on that sub-question gets penalised. Note this
is *not* "Phase 1 is worse on formoterol overall" — Phase 1 is better
on sentences 1-2. It's specifically sentences 4-5 that drive the loss,
and the cause is documented: the reranker demoted small-population
specifics in favour of general topical relevance.

## Reading the three topics together

Three patterns, mechanically selected and qualitatively interpretable:

1. **Phase 1 wins** when the question is non-classical and the MedCPT-CE
   rerank surfaces PMIDs the LLM judge ratifies but the human pool
   never saw (qa=150).
2. **Tied** when both pipelines converge on the same topical
   understanding but disagree on which specific documents to cite —
   both ratified by the LLM judge on disjoint sets (qa=120).
3. **Phase 1 loses** when the topic has a well-curated classical
   literature in the human pool that BM25's lexical bias naturally
   surfaces and that the reranker actively demotes (qa=131).

The aggregate Supports F1 of 16.43 (Phase 1) vs 16.55 (starter) on the
intersection pool is the net of these effects: roughly balanced
trade-offs, with neither pipeline systematically better on supports.
The pool-aware methodology is essential to see this — on the official
pool alone, starter scored 44.34 and Phase 1 scored 5.55, and the gap
was largely artefactual.

## Reproducibility appendix — full sorted topic-Δ list

Strict Supports F1 on intersection pool, target = `phase1_baseline`,
anchor = `starter_baseline`. Topics with identical Δ are tie-broken by
ascending integer qa_id, so re-running `--select-3` always yields the
same three picks.

| qa_id | Δ F1 |
|---|---|
| 150 | +0.1394 |
| 139 | +0.1370 |
| 149 | +0.1043 |
| 138 | +0.1042 |
| 152 | +0.0983 |
| 141 | +0.0668 |
| 128 | +0.0582 |
| 117 | +0.0552 |
| 118 | +0.0544 |
| 127 | +0.0500 |
| 137 | +0.0438 |
| 140 | +0.0433 |
| 146 | +0.0381 |
| 151 | +0.0350 |
| 121 | +0.0257 |
| 154 | +0.0241 |
| 136 | +0.0147 |
| 119 | +0.0122 |
| 147 | +0.0101 |
| 134 | +0.0059 |
| 132 | +0.0000 |
| 120 | +0.0000 |
| 144 | -0.0009 |
| 116 | -0.0069 |
| 133 | -0.0098 |
| 124 | -0.0148 |
| 135 | -0.0155 |
| 155 | -0.0251 |
| 142 | -0.0301 |
| 130 | -0.0318 |
| 123 | -0.0401 |
| 153 | -0.0433 |
| 145 | -0.0455 |
| 129 | -0.0501 |
| 122 | -0.0644 |
| 148 | -0.0758 |
| 125 | -0.1142 |
| 126 | -0.1188 |
| 143 | -0.1391 |
| 131 | -0.1949 |
