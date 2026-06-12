from data.build_evidence_corpus import build_evidence_corpus
from data.quality_audit import quality_audit
from data.schemas import ClaimEvidenceRecord, EvidenceSpan


def test_quality_audit_reports_expected_counts(tmp_path) -> None:
    records = [
        ClaimEvidenceRecord(
            claim_id="1",
            claim="The sky is blue.",
            label="SUPPORTED",
            split="train",
            evidence=(EvidenceSpan(doc_id="d1", text="The sky is blue."),),
        ),
        ClaimEvidenceRecord(
            claim_id="2",
            claim="The sky is blue.",
            label="REFUTED",
            split="validation",
            evidence=(),
        ),
    ]

    report = quality_audit(records)

    assert report["label_distribution"] == {"SUPPORTED": 1, "REFUTED": 1}
    assert report["duplicate_count"] == 1
    assert report["missing_evidence_count"] == 1
    assert report["split_stats"] == {"train": 1, "validation": 1}

    output = build_evidence_corpus(records, tmp_path / "corpus.jsonl")
    assert output.exists()
    assert output.read_text(encoding="utf-8").count("\n") == 1
