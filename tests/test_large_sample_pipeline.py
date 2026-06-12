"""Unit tests for the large-sample build helpers (no network/downloads)."""

from __future__ import annotations

import json

from data.schemas import ClaimEvidenceRecord, EvidenceSpan
from scripts.build_large_sample_datasets import (
    assemble_evidence_corpus,
    corpus_rows_from_spans,
    split_records,
    write_corpus_rows,
)


def _record(claim_id: str, label: str, spans: list[EvidenceSpan]) -> ClaimEvidenceRecord:
    return ClaimEvidenceRecord(claim_id=claim_id, claim=f"claim {claim_id}", label=label, evidence=tuple(spans))


def test_split_records_is_deterministic_and_disjoint() -> None:
    pool = [_record(str(i), "SUPPORTED", []) for i in range(10)]
    val_a, test_a = split_records(pool, val_size=4, seed=42)
    val_b, test_b = split_records(pool, val_size=4, seed=42)

    assert [r.claim_id for r in val_a] == [r.claim_id for r in val_b]
    assert len(val_a) == 4
    assert len(test_a) == 6
    val_ids = {r.claim_id for r in val_a}
    test_ids = {r.claim_id for r in test_a}
    assert val_ids.isdisjoint(test_ids)
    assert val_ids | test_ids == {str(i) for i in range(10)}


def test_split_records_clamps_oversized_val() -> None:
    pool = [_record(str(i), "REFUTED", []) for i in range(3)]
    val, test = split_records(pool, val_size=99, seed=1)
    assert len(val) == 3
    assert test == []


def test_corpus_rows_dedupe_and_skip_empty() -> None:
    spans = [
        EvidenceSpan(doc_id="a", text="first abstract"),
        EvidenceSpan(doc_id="a", text="duplicate id ignored"),
        EvidenceSpan(doc_id="b", text="   "),  # empty text skipped
        EvidenceSpan(doc_id="c", text="third abstract", title="T"),
    ]
    rows = corpus_rows_from_spans(spans, source_split="scifact_corpus")
    doc_ids = [row["doc_id"] for row in rows]
    assert doc_ids == ["a", "c"]
    assert rows[0]["split"] == "scifact_corpus"
    assert rows[1]["title"] == "T"


def test_assemble_corpus_includes_distractors_and_fever_evidence() -> None:
    scifact_corpus = [
        EvidenceSpan(doc_id="sci1", text="relevant abstract"),
        EvidenceSpan(doc_id="sci2", text="distractor abstract"),
    ]
    fever_records = [
        _record("f1", "SUPPORTED", [EvidenceSpan(doc_id="Page:0", text="fever sentence")]),
        _record("f2", "NOT_ENOUGH_INFO", []),
    ]
    rows = assemble_evidence_corpus(scifact_corpus, fever_records)
    doc_ids = {row["doc_id"] for row in rows}
    assert {"sci1", "sci2", "Page:0"} <= doc_ids
    # full scifact corpus acts as distractors -> more than just the relevant docs
    assert len(rows) == 3


def test_write_corpus_rows_roundtrip(tmp_path) -> None:
    rows = [{"doc_id": "a", "text": "x", "title": None, "split": "s", "metadata": {}}]
    path = write_corpus_rows(rows, tmp_path / "corpus.jsonl")
    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert loaded[0]["doc_id"] == "a"
