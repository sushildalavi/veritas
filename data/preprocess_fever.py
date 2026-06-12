"""FEVER preprocessing helpers."""

from __future__ import annotations

from .schemas import ClaimEvidenceRecord, EvidenceSpan


def normalize_fever_row(row: dict) -> ClaimEvidenceRecord:
    evidence_texts = row.get("evidence", []) or []
    evidence = tuple(
        EvidenceSpan(doc_id=str(index), text=str(text), title=row.get("title"))
        for index, text in enumerate(evidence_texts)
    )
    return ClaimEvidenceRecord(
        claim_id=str(row.get("id", row.get("claim_id", ""))),
        claim=str(row.get("claim", "")),
        label=str(row.get("label", "NOT ENOUGH INFO")),
        evidence=evidence,
        split=row.get("split"),
        metadata={"source": "fever"},
    )
