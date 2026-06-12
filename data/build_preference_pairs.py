"""Placeholder preference-pair builder for later alignment stages."""

from __future__ import annotations

from .schemas import ClaimEvidenceRecord


def build_preference_pairs(records: list[ClaimEvidenceRecord]) -> list[dict[str, object]]:
    """Create a minimal placeholder structure for future preference data."""

    pairs: list[dict[str, object]] = []
    for record in records:
        if len(record.evidence) < 2:
            continue
        pairs.append(
            {
                "claim_id": record.claim_id,
                "chosen": record.evidence[0].text,
                "rejected": record.evidence[1].text,
                "label": record.label,
            }
        )
    return pairs
