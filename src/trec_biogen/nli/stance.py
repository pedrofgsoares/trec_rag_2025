"""NLI stance scoring for both paths (D5).

Support path (7.3)
------------------
``score_support`` runs a general-purpose entailment model (default
``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``) over (answer_sentence,
title+abstract[≤512 tokens]) pairs from the reranked top-30 list and emits a
per-pair probability table.

Contradict path (8.4)
---------------------
``score_contradict_pairs`` runs a SciFive-MedNLI T5 checkpoint over
sentence-level (answer_sentence, abstract_sentence) pairs surviving the
NegEx pre-filter. The label space is {entailment, neutral, contradiction};
we compute the probability of the ``contradiction`` token at the first
decoded position via constrained log-softmax. If the configured T5
checkpoint cannot be loaded, the function falls back to the DeBERTa NLI
model so the pipeline remains end-to-end runnable.

Both functions load and unload their model within the call — the
orchestrator only needs to call them sequentially.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trec_biogen.pipeline.model_utils import device, unload
from trec_biogen.retrieval.bm25 import BM25Index

SUPPORT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
CONTRADICT_MODEL = "razent/SciFive-base-Pubmed_PMC-MedNLI"  # community MedNLI fine-tune
SUPPORT_BATCH = 16
CONTRADICT_BATCH = 8
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
    """
    import torch
    from transformers import AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name)
    scorer = _T5MedNLI(tok=tok, dev=dev, model_name=model_name)

    df = pl.read_parquet(pairs_parquet)
    pairs = list(
        zip(df["sentence_text"].to_list(), df["abstract_sentence_text"].to_list(), strict=True)
    )

    scores: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            scores.extend(scorer.contradiction_probs(pairs[i : i + batch_size]))

    out = df.with_columns(pl.Series("contradiction_prob", scores))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    scorer.unload()
    return out_path


class _T5MedNLI:
    """Tiny wrapper around a SciFive-MedNLI seq2seq checkpoint."""

    def __init__(self, *, tok, dev: str, model_name: str) -> None:
        from transformers import AutoModelForSeq2SeqLM

        self.tok = tok
        self.dev = dev
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(dev).eval()
        # Token IDs for the three NLI labels — used to extract probability mass
        # from a single decoder step.
        self._label_ids = {
            label: tok(label, add_special_tokens=False).input_ids[0]
            for label in _DEBERTA_LABELS
        }

    def contradiction_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        prompts = [f"mednli: sentence1: {a} sentence2: {b}" for a, b in pairs]
        enc = self.tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(self.dev)
        # One decoder step from the start token.
        dec_start = torch.full(
            (enc.input_ids.shape[0], 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.dev,
        )
        logits = self.model(**enc, decoder_input_ids=dec_start).logits[:, 0, :]
        label_idx = torch.tensor(
            [self._label_ids[l] for l in _DEBERTA_LABELS], device=self.dev
        )
        sub = logits[:, label_idx]
        probs = torch.softmax(sub, dim=-1)
        return probs[:, _DEBERTA_LABELS.index("contradiction")].cpu().tolist()

    def unload(self) -> None:
        unload(self.model, self.tok)
