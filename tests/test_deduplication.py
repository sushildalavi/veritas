from data.deduplicate import NearDuplicateDetector, deduplicate_records, exact_deduplicate
from data.schemas import ClaimEvidenceRecord, EvidenceSpan


def test_exact_deduplicate_preserves_order() -> None:
    spans = [
        EvidenceSpan(doc_id="1", text="Alpha"),
        EvidenceSpan(doc_id="2", text="Alpha"),
        EvidenceSpan(doc_id="3", text="Beta"),
    ]

    deduped = exact_deduplicate(spans, key="text")

    assert [span.text for span in deduped] == ["Alpha", "Beta"]


def test_deduplicate_records_removes_duplicate_claims_and_evidence() -> None:
    records = [
        ClaimEvidenceRecord(
            claim_id="1",
            claim="The sky is blue.",
            label="SUPPORTED",
            evidence=(
                EvidenceSpan(doc_id="a", text="The sky is blue."),
                EvidenceSpan(doc_id="b", text="The sky is blue."),
            ),
        ),
        ClaimEvidenceRecord(
            claim_id="2",
            claim="The sky is blue.",
            label="SUPPORTED",
            evidence=(EvidenceSpan(doc_id="c", text="Blue sky example."),),
        ),
    ]

    deduped = deduplicate_records(records)

    assert len(deduped) == 1
    assert len(deduped[0].evidence) == 1


def test_near_duplicate_detector_uses_token_overlap() -> None:
    detector = NearDuplicateDetector(threshold=0.5)

    assert detector.is_near_duplicate("alpha beta", "beta alpha")
