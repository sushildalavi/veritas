from pathlib import Path

from data.sample_pipeline import (
    _normalize_fever_row,
    _normalize_scifact_row,
    build_data_quality_markdown,
    normalize_sample_label,
    write_sample_jsonl,
)
from data.schemas import EvidenceSpan


def test_normalize_sample_label_maps_common_variants() -> None:
    assert normalize_sample_label("supports") == "SUPPORTED"
    assert normalize_sample_label("REFUTES") == "REFUTED"
    assert normalize_sample_label("not enough info") == "NOT_ENOUGH_INFO"


def test_write_sample_jsonl_round_trips(tmp_path: Path) -> None:
    records = [
        _dummy_record(),
    ]
    path = write_sample_jsonl(records, tmp_path / "out.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"claim": "A claim"' in lines[0]


def test_build_data_quality_markdown_contains_metrics() -> None:
    md = build_data_quality_markdown(
        {
            "label_distribution": {"SUPPORTED": 1},
            "duplicate_count": 0,
            "missing_evidence_count": 0,
            "average_claim_length": 3.0,
            "average_evidence_length": 7.0,
            "split_stats": {"train": 1},
        }
    )
    assert "data quality report" in md.lower()
    assert "label_distribution" in md


def test_normalize_fever_row_uses_fallback_sentence(monkeypatch) -> None:
    monkeypatch.setattr("data.sample_pipeline._fetch_wikipedia_extract", lambda title: "First sentence. Second sentence. Third sentence.")
    row = {
        "id": 1,
        "label": "SUPPORTS",
        "claim": "Test claim",
        "evidence": [[[1, 2, "Some_Page", 1]]],
    }
    record = _normalize_fever_row(row)
    assert record.label == "SUPPORTED"
    assert record.evidence[0].text == "Second sentence."


def test_normalize_scifact_row_builds_text_from_abstract() -> None:
    corpus = [EvidenceSpan(doc_id="10", text="Sentence one. Sentence two. Sentence three.", title="Paper")]
    row = {
        "id": 5,
        "claim": "A claim",
        "evidence": {"10": [{"label": "SUPPORT", "sentences": [0, 2]}]},
        "cited_doc_ids": [10],
    }
    record = _normalize_scifact_row(row, corpus)
    assert record.label == "SUPPORTED"
    assert record.evidence[0].text == "Sentence one. Sentence three."


def _dummy_record():
    from data.schemas import ClaimEvidenceRecord

    return ClaimEvidenceRecord(
        claim_id="1",
        claim="A claim",
        label="SUPPORTED",
        evidence=(EvidenceSpan(doc_id="1", text="Evidence text"),),
    )
