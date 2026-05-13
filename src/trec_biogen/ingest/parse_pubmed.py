"""Parse the BioGen 2025 PubMed corpus into a flat JSONL stream.

Output record schema (one line per article):
    {
      "pmid": "12345678",
      "title": "...",
      "abstract": "...",
      "mesh": ["D012345", ...],
      "pubdate": "2023-04",
      "journal": "Nature",
      "empty_abstract": false
    }

Supported inputs:
  * NLM PubMed baseline XML (``pubmed*n*.xml.gz`` files) — the canonical
    distribution form.
  * Pre-parsed JSONL (one PubmedArticle-shaped record per line) — pass-through
    used by tests and by alternative starter-kit distributions.

Task: 3.3, 3.4, 3.5
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any
from xml.etree import ElementTree as ET

import orjson


def _iter_articles_xml(stream: IO[bytes]) -> Iterator[ET.Element]:
    """Yield each ``<PubmedArticle>`` element from a streamed XML file.

    Uses iterparse so peak memory stays bounded regardless of file size.
    """
    context = ET.iterparse(stream, events=("end",))
    for _, elem in context:
        if elem.tag == "PubmedArticle":
            yield elem
            elem.clear()


def _text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    # itertext joins inline markup (<i>, <sub>, ...) into a single string.
    return "".join(elem.itertext()).strip()


def _parse_pubdate(article: ET.Element) -> str:
    """Return ``YYYY`` or ``YYYY-MM`` from PubDate, best-effort."""
    pd = article.find(".//Journal/JournalIssue/PubDate")
    if pd is None:
        return ""
    year = _text(pd.find("Year"))
    if not year:
        # MedlineDate fallback, e.g. "2003 Spring"
        return _text(pd.find("MedlineDate"))[:7]
    month = _text(pd.find("Month"))
    if not month:
        return year
    # Normalise three-letter months to numbers when possible.
    months = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
        "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }
    return f"{year}-{months.get(month[:3], month[:2].zfill(2))}"


def _parse_mesh(citation: ET.Element) -> list[str]:
    out: list[str] = []
    for mh in citation.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
        ui = mh.get("UI")
        if ui:
            out.append(ui)
    return out


def _record_from_pubmed_article(pa: ET.Element) -> dict[str, Any]:
    citation = pa.find("MedlineCitation")
    if citation is None:
        return {}
    article = citation.find("Article")
    if article is None:
        return {}
    pmid = _text(citation.find("PMID"))
    title = _text(article.find("ArticleTitle"))
    # Abstract may have multiple <AbstractText Label="..."> sections.
    abstract_parts = [
        ((seg.get("Label") + ": ") if seg.get("Label") else "") + _text(seg)
        for seg in article.findall(".//Abstract/AbstractText")
    ]
    abstract = " ".join(p for p in abstract_parts if p).strip()
    journal = _text(article.find(".//Journal/Title"))
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "mesh": _parse_mesh(citation),
        "pubdate": _parse_pubdate(article),
        "journal": journal,
        "empty_abstract": not bool(abstract),
    }


def _open_xml(path: Path) -> IO[bytes]:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _iter_xml_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        if root.suffix in {".gz", ".xml"}:
            yield root
        return
    for pattern in ("*.xml.gz", "*.xml"):
        yield from sorted(root.rglob(pattern))


def iter_records(input_path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records from any supported input form under ``input_path``."""
    # Pass-through for pre-parsed JSONL (test fixtures, alt distributions).
    if input_path.is_file() and input_path.suffix in {".jsonl", ".json"}:
        with input_path.open("rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield orjson.loads(line)
        return

    for xml_path in _iter_xml_files(input_path):
        with _open_xml(xml_path) as fh:
            for pa in _iter_articles_xml(fh):
                rec = _record_from_pubmed_article(pa)
                if rec.get("pmid"):
                    yield rec


def write_jsonl(records: Iterable[dict[str, Any]], out_path: Path) -> int:
    """Write records to ``out_path`` (one JSON object per line). Returns count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("wb") as fh:
        for rec in records:
            fh.write(orjson.dumps(rec))
            fh.write(b"\n")
            n += 1
    return n


def count_records(input_path: Path) -> int:
    return sum(1 for _ in iter_records(input_path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Parse BioGen PubMed corpus to JSONL.")
    p.add_argument("--input", required=True, type=Path, help="XML dir or JSONL file")
    p.add_argument("--output", type=Path, help="Output JSONL path")
    p.add_argument("--count", action="store_true", help="Print doc count and exit")
    args = p.parse_args(argv)

    if args.count:
        print(count_records(args.input))
        return 0

    if not args.output:
        p.error("--output is required unless --count is set")
    n = write_jsonl(iter_records(args.input), args.output)
    print(f"wrote {n} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
