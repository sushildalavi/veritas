"""Build grounded context from ranked evidence."""

from __future__ import annotations

from dataclasses import dataclass

from data.schemas import EvidenceSpan


@dataclass(frozen=True)
class ContextEvidence:
    citation_id: int
    evidence: EvidenceSpan


@dataclass(frozen=True)
class ContextBundle:
    claim: str
    evidence_items: tuple[ContextEvidence, ...]

    def format_block(self) -> str:
        return "\n".join(
            f"[{item.citation_id}] {item.evidence.title + ': ' if item.evidence.title else ''}{item.evidence.text}"
            for item in self.evidence_items
        )


def build_context(claim: str, ranked_evidence: list[EvidenceSpan], top_k: int = 5) -> ContextBundle:
    items = tuple(
        ContextEvidence(citation_id=index + 1, evidence=span)
        for index, span in enumerate(ranked_evidence[:top_k])
    )
    return ContextBundle(claim=claim, evidence_items=items)
