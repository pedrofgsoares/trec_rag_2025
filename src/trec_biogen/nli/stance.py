"""NLI stance scoring for both paths.

Both paths use a 3-way NLI classifier (default
``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``); the only difference is
the input granularity and which probability column we read downstream:

* **Support path** — passages are (answer_sentence, title+abstract[≤512]).
  We read ``entailment_prob`` for ranking. Task 7.3.
* **Contradict path** — passages are (answer_sentence, abstract_sentence)
  surviving the NegEx pre-filter. We read ``contradiction_prob`` for
  ranking. Task 8.4.

The original design D5 called for ``razent/SciFive-base-Pubmed_PMC-MedNLI``
on the contradict path, but no such model exists on the Hub (only the
``-large`` variant is published). Phase 1 uses DeBERTa-MNLI for both paths;
Phase 2 may revisit with SciFive-large fp16 if biomedical specialisation
proves necessary.

Each function loads and unloads its model within the call — the
orchestrator only needs to call them sequentially.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trec_biogen.pipeline.model_utils import device, unload
from trec_biogen.retrieval.bm25 import BM25Index

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
SUPPORT_MODEL = DEFAULT_NLI_MODEL
CONTRADICT_MODEL = DEFAULT_NLI_MODEL
SUPPORT_BATCH = 16
CONTRADICT_BATCH = 16
MAX_LEN = 512

# DeBERTa MNLI label order is: entailment(0), neutral(1), contradiction(2).
_DEBERTA_LABELS = ("entailment", "neutral", "contradiction")


def score_support(
    rerank_parquet: Path,
    bm25: BM25Index,
    *,
    out_path: Path,
    model_name: str = SUPPORT_MODEL,
    batch_size: int = SUPPORT_BATCH,
) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()

    df = pl.read_parquet(rerank_parquet)
    pmids = df["candidate_pmid"].unique().to_list()
    doc_text = {pmid: bm25.doc_text(pmid) for pmid in pmids}

    rows = df.to_dicts()
    pairs = [(r["sentence_text"], doc_text.get(r["candidate_pmid"], "")) for r in rows]

    ent, neu, con = [], [], []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = tok.batch_encode_plus(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(dev)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().tolist()
            for p in probs:
                ent.append(p[0])
                neu.append(p[1])
                con.append(p[2])

    out = (
        pl.DataFrame(rows)
        .with_columns(
            pl.Series("entailment_prob", ent),
            pl.Series("neutral_prob", neu),
            pl.Series("contradiction_prob", con),
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    unload(model, tok)
    return out_path


def score_contradict_pairs(
    pairs_parquet: Path,
    *,
    out_path: Path,
    model_name: str = CONTRADICT_MODEL,
    batch_size: int = CONTRADICT_BATCH,
) -> Path:
    """Score (answer_sentence, abstract_sentence) pairs for contradiction.

    Expected input columns: ``qa_id, sentence_id, candidate_pmid,
    abstract_sentence_idx, abstract_sentence_text, sentence_text, bm25_rank,
    bm25_score``.

    Uses a sequence-classification 3-way NLI model and returns the
    contradiction-class probability per pair.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()

    df = pl.read_parquet(pairs_parquet)
    pairs = list(
        zip(df["sentence_text"].to_list(), df["abstract_sentence_text"].to_list(), strict=True)
    )

    con_idx = _DEBERTA_LABELS.index("contradiction")
    scores: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = tok.batch_encode_plus(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(dev)
            probs = torch.softmax(model(**enc).logits, dim=-1)
            scores.extend(probs[:, con_idx].cpu().tolist())

    out = df.with_columns(pl.Series("contradiction_prob", scores))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    unload(model, tok)
    return out_path
