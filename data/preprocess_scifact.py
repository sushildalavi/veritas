"""SciFact preprocessing helpers."""

from __future__ import annotations

from .schemas import ClaimEvidenceRecord, EvidenceSpan


def normalize_scifact_row(row: dict) -> ClaimEvidenceRecord:
    evidence_items = row.get("evidence", []) or []
    evidence = tuple(
        EvidenceSpan(
            doc_id=str(item.get("doc_id", index)),
            text=str(item.get("text", "")),
            title=item.get("title"),
        )
        for index, item in enumerate(evidence_items)
    )
    return ClaimEvidenceRecord(
        claim_id=str(row.get("id", row.get("claim_id", ""))),
        claim=str(row.get("claim", "")),
        label=str(row.get("label", "NOT ENOUGH INFO")),
        evidence=evidence,
        split=row.get("split"),
        metadata={"source": "scifact"},
    )
