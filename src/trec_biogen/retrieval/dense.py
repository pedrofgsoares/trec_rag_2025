"""FAISS-backed dense retrieval over the MedCPT-Encoder 5M-doc subset.

Phase 2 §9.3. Encodes the query online with ``MedCPT-Query-Encoder``
(BERT-base, 109 M params, CPU-friendly), runs a FAISS-CPU search against
the pre-built ``data/indexes/medcpt_5m/index.faiss``, and returns ranked
PMIDs by looking up the row index in ``pmid_lookup.parquet``.

The index is built once offline by ``scripts/build_dense_index.sh`` (§9.2)
over the top-5M PMIDs selected by ``scripts/select_top5m_pmids.py`` (§9.1).
At search time the article encoder is *not* needed — only the query
encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import faiss
    import torch

DEFAULT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
DEFAULT_INDEX_DIR = Path("data/indexes/medcpt_5m")
DEFAULT_MAX_LEN = 64   # MedCPT-Query-Encoder is trained on short PubMed queries


@dataclass(slots=True, frozen=True)
class DenseHit:
    pmid: str
    rank: int
    score: float


class DenseIndex:
    """Open the FAISS-CPU MedCPT index and serve dense ranked lists.

    Constructor loads:
      * the FAISS index (``index.faiss``),
      * the row → PMID lookup (``pmid_lookup.parquet``),
      * the query encoder (BERT-base, CPU).

    The query encoder dominates wall-clock per query (~50 ms on a modern
    CPU at ``max_len=64``); FAISS search itself is sub-millisecond at
    5M docs.
    """

    def __init__(
        self,
        index_dir: Path | str,
        *,
        query_model: str = DEFAULT_QUERY_MODEL,
        max_len: int = DEFAULT_MAX_LEN,
    ) -> None:
        import faiss
        from transformers import AutoModel, AutoTokenizer

        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(f"dense index dir not found: {self.index_dir}")
        faiss_path = self.index_dir / "index.faiss"
        lookup_path = self.index_dir / "pmid_lookup.parquet"
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index missing: {faiss_path}")
        if not lookup_path.exists():
            raise FileNotFoundError(f"pmid_lookup missing: {lookup_path}")

        self._index: faiss.Index = faiss.read_index(str(faiss_path))
        lookup = pl.read_parquet(lookup_path)
        # Convention: row i in the FAISS index corresponds to lookup.pmid[i].
        self._pmids: list[str] = [str(p) for p in lookup["pmid"].to_list()]
        if self._index.ntotal != len(self._pmids):
            raise ValueError(
                f"FAISS index has {self._index.ntotal} vectors but lookup has "
                f"{len(self._pmids)} pmids — index and lookup are out of sync."
            )

        self._tok = AutoTokenizer.from_pretrained(query_model)
        self._model = AutoModel.from_pretrained(query_model).eval()
        self._max_len = max_len

    def encode(self, query_text: str):
        """Return the L2-normalised query embedding as a (1, dim) numpy array.

        MedCPT is trained with cosine similarity, and the FAISS index is
        built as ``IndexFlatIP`` over normalised vectors — so we normalise
        the query here too.
        """
        import numpy as np
        import torch

        enc = self._tok(
            query_text,
            padding=True,
            truncation=True,
            max_length=self._max_len,
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = self._model(**enc)
        # MedCPT-Query-Encoder uses the [CLS] token's last hidden state.
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        emb /= max(1e-12, float(np.linalg.norm(emb)))
        return emb

    def search(self, query_text: str, k: int) -> list[DenseHit]:
        """Run dense retrieval, return up to ``k`` ranked PMIDs."""
        emb = self.encode(query_text)
        scores, indices = self._index.search(emb, k)
        out: list[DenseHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0].tolist(), indices[0].tolist())):
            if idx < 0:
                continue  # FAISS pads with -1 when fewer than k results
            out.append(DenseHit(pmid=self._pmids[idx], rank=rank + 1, score=float(score)))
        return out

    def close(self) -> None:
        self._index = None  # type: ignore[assignment]
        self._model = None  # type: ignore[assignment]


def verify_index(index_dir: Path | str, *, min_docs: int = 1_000_000) -> None:
    """Preflight: open the dense index and assert it looks like the 5M-subset.

    Raises ``RuntimeError`` if the index is missing, the lookup has
    fewer than ``min_docs`` rows, or the two are out of sync.
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise RuntimeError(f"dense index missing: {index_dir} (build via scripts/build_dense_index.sh)")
    di = DenseIndex(index_dir)
    n = len(di._pmids)
    di.close()
    if n < min_docs:
        raise RuntimeError(
            f"dense index has {n} docs, expected ≥ {min_docs}. Did encoding complete?"
        )
