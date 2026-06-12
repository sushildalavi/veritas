"""Quality audit utilities for the fact verification corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from statistics import mean

from .schemas import ClaimEvidenceRecord


def quality_audit(records: list[ClaimEvidenceRecord]) -> dict[str, object]:
    label_counts = Counter(record.label for record in records)
    duplicate_claims = len(records) - len({record.claim for record in records})
    missing_evidence = sum(1 for record in records if not record.evidence)
    claim_lengths = [len(record.claim.split()) for record in records if record.claim]
    evidence_lengths = [
        len(span.text.split())
        for record in records
        for span in record.evidence
        if span.text
    ]
    split_counts = Counter(record.split or "unspecified" for record in records)

    return {
        "label_distribution": dict(label_counts),
        "duplicate_count": duplicate_claims,
        "missing_evidence_count": missing_evidence,
        "average_claim_length": mean(claim_lengths) if claim_lengths else 0.0,
        "average_evidence_length": mean(evidence_lengths) if evidence_lengths else 0.0,
        "split_stats": dict(split_counts),
        "sample_record": asdict(records[0]) if records else None,
    }
