"""Schema + empty_abstract flag tests for the PubMed parser. Task 3.4."""

from __future__ import annotations

import json
from pathlib import Path

from trec_biogen.ingest.parse_pubmed import (
    count_records,
    iter_records,
    write_jsonl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pubmed_mini.xml"
REQUIRED_KEYS = {"pmid", "title", "abstract", "mesh", "pubdate", "journal", "empty_abstract"}


def test_schema_keys_present() -> None:
    for rec in iter_records(FIXTURE):
        assert REQUIRED_KEYS <= set(rec), f"missing keys: {REQUIRED_KEYS - set(rec)}"
        assert isinstance(rec["pmid"], str) and rec["pmid"]
        assert isinstance(rec["mesh"], list)
        assert isinstance(rec["empty_abstract"], bool)


def test_empty_abstract_flag() -> None:
    by_pmid = {r["pmid"]: r for r in iter_records(FIXTURE)}
    assert by_pmid["10000001"]["empty_abstract"] is False
    assert by_pmid["10000002"]["empty_abstract"] is True
    assert by_pmid["10000003"]["empty_abstract"] is False


def test_abstract_concatenates_labeled_sections() -> None:
    by_pmid = {r["pmid"]: r for r in iter_records(FIXTURE)}
    abs1 = by_pmid["10000001"]["abstract"]
    assert "BACKGROUND:" in abs1
    assert "RESULTS:" in abs1
    assert "Aspirin reduced MI by 25%" in abs1


def test_pubdate_normalisation() -> None:
    by_pmid = {r["pmid"]: r for r in iter_records(FIXTURE)}
    assert by_pmid["10000001"]["pubdate"] == "2021-03"
    assert by_pmid["10000003"]["pubdate"] == "2019"
    assert by_pmid["10000002"]["pubdate"].startswith("2003")


def test_mesh_descriptor_uis() -> None:
    by_pmid = {r["pmid"]: r for r in iter_records(FIXTURE)}
    assert by_pmid["10000001"]["mesh"] == ["D001241", "D002318"]
    assert by_pmid["10000002"]["mesh"] == []


def test_count_records() -> None:
    assert count_records(FIXTURE) == 3


def test_write_jsonl_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    n = write_jsonl(iter_records(FIXTURE), out)
    assert n == 3
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    # Pass-through on JSONL re-read.
    reread = list(iter_records(out))
    assert {r["pmid"] for r in reread} == {"10000001", "10000002", "10000003"}
    # Lines are valid JSON.
    for line in lines:
        json.loads(line)
