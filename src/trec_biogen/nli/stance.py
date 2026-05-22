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


def _t5_label_token_ids(tokenizer) -> dict[str, int]:
    """First-token ID for each MedNLI label, used for constrained decoding.

    SciFive-large outputs the label as a free-form string ("entailment",
    "neutral", "contradiction"). At inference we only generate one token
    and project the logits onto these three IDs to extract calibrated
    probabilities — equivalent to a classifier head, in a single
    decoding step.
    """
    out: dict[str, int] = {}
    for label in _DEBERTA_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if not ids:
            raise ValueError(f"tokenizer produced no tokens for label {label!r}")
        out[label] = ids[0]
    return out


def score_contradict_pairs_t5(
    pairs_parquet: Path,
    *,
    out_path: Path,
    model_name: str,
    batch_size: int = 4,
    fp16: bool = True,
    chunk_size: int = 4,
) -> Path:
    """T5 seq2seq variant of the contradict NLI step (Phase 2 §7).

    Designed for ``razent/SciFive-large-Pubmed_PMC-MedNLI``: feeds the
    standard MedNLI prompt format and uses constrained decoding over the
    three label tokens to extract ``contradiction_prob`` per pair.

    Same input/output schema as :func:`score_contradict_pairs` so the
    downstream aggregator is identical. ``chunk_size`` caps the per-step
    forward batch for memory; on 4 GB VRAM with fp16 the safe value is 4.
    ``batch_size`` is the outer iteration unit and may be > ``chunk_size``
    (it is then split into chunks internally).
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if fp16 else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=dtype,
    ).to(dev).eval()

    df = pl.read_parquet(pairs_parquet)
    pairs = list(
        zip(df["sentence_text"].to_list(), df["abstract_sentence_text"].to_list(), strict=True)
    )

    label_ids = _t5_label_token_ids(tok)
    id_order = [label_ids[lbl] for lbl in _DEBERTA_LABELS]
    con_local_idx = _DEBERTA_LABELS.index("contradiction")

    chunk = max(1, min(chunk_size, batch_size))
    scores: list[float] = []
    with torch.inference_mode():
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start : batch_start + batch_size]
            for sub_start in range(0, len(batch), chunk):
                sub = batch[sub_start : sub_start + chunk]
                prompts = [
                    f"mednli: premise: {abstract} hypothesis: {sent}"
                    for sent, abstract in sub
                ]
                enc = tok.batch_encode_plus(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                ).to(dev)
                gen = model.generate(
                    **enc,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                first_step_logits = gen.scores[0]
                label_logits = first_step_logits[:, id_order]
                probs = torch.softmax(label_logits.float(), dim=-1)
                scores.extend(probs[:, con_local_idx].cpu().tolist())

    out = df.with_columns(pl.Series("contradiction_prob", scores))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    unload(model, tok)
    return out_path
