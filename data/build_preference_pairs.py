"""Build preference pairs for offline alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateExplanation:
    claim_id: str
    explanation: str
    verdict_consistency: float
    citation_precision: float
    faithfulness: float

    @property
    def quality_score(self) -> float:
        return 0.4 * self.verdict_consistency + 0.3 * self.citation_precision + 0.3 * self.faithfulness


def score_candidate(candidate: CandidateExplanation | dict[str, Any]) -> float:
    if isinstance(candidate, dict):
        return CandidateExplanation(
            claim_id=str(candidate["claim_id"]),
            explanation=str(candidate["explanation"]),
            verdict_consistency=float(candidate.get("verdict_consistency", 0.0)),
            citation_precision=float(candidate.get("citation_precision", 0.0)),
            faithfulness=float(candidate.get("faithfulness", 0.0)),
        ).quality_score
    return candidate.quality_score


def build_preference_pairs(
    candidate_groups: list[list[CandidateExplanation | dict[str, Any]]],
    *,
    quality_gap_threshold: float = 0.2,
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for group in candidate_groups:
        normalized = [
            candidate
            if isinstance(candidate, CandidateExplanation)
            else CandidateExplanation(
                claim_id=str(candidate["claim_id"]),
                explanation=str(candidate["explanation"]),
                verdict_consistency=float(candidate.get("verdict_consistency", 0.0)),
                citation_precision=float(candidate.get("citation_precision", 0.0)),
                faithfulness=float(candidate.get("faithfulness", 0.0)),
            )
            for candidate in group
        ]
        if len(normalized) < 2:
            continue
        ranked = sorted(normalized, key=lambda item: item.quality_score, reverse=True)
        chosen, rejected = ranked[0], ranked[-1]
        if chosen.quality_score - rejected.quality_score < quality_gap_threshold:
            continue
        pairs.append(
            {
                "claim_id": chosen.claim_id,
                "chosen": chosen.explanation,
                "rejected": rejected.explanation,
                "chosen_score": chosen.quality_score,
                "rejected_score": rejected.quality_score,
            }
        )
    return pairs
