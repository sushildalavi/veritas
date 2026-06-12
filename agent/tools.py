"""Composable tools for the reflection loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from data.schemas import EvidenceSpan
from models.deberta_verifier import VerificationResult
from rag import CitationCheckResult, ContextBundle, build_context, check_citations, generate_explanation


@dataclass(frozen=True)
class RetrievalResult:
    evidence: list[EvidenceSpan]


def retrieve_evidence(retriever: object, claim: str, top_k: int = 5) -> RetrievalResult:
    return RetrievalResult(evidence=list(retriever.retrieve(claim, top_k=top_k)))


def rank_evidence(ranker: object | None, claim: str, evidence: list[EvidenceSpan]) -> list[EvidenceSpan]:
    if ranker is None:
        return evidence
    try:
        return list(ranker.rank(claim, evidence))
    except Exception:
        return evidence


def verify_claim(verifier: object, claim: str, evidence: list[EvidenceSpan]) -> VerificationResult:
    return verifier.predict(claim, evidence)


def build_grounded_context(claim: str, evidence: list[EvidenceSpan], top_k: int = 5) -> ContextBundle:
    return build_context(claim, evidence, top_k=top_k)


def explain_claim(
    context: ContextBundle,
    verification: VerificationResult,
    generator: Callable[[str], str] | None = None,
):
    return generate_explanation(context, verification, generator=generator)


def validate_citations(explanation: str, context: ContextBundle) -> CitationCheckResult:
    return check_citations(explanation, context)
