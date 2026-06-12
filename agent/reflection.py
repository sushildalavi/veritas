"""State machine for self-checking claim verification."""

from __future__ import annotations

from dataclasses import dataclass, field

from data.schemas import EvidenceSpan
from models.deberta_verifier import VerificationResult
from .tools import build_grounded_context, explain_claim, rank_evidence, retrieve_evidence, validate_citations, verify_claim


@dataclass(frozen=True)
class ReflectionOutcome:
    decision: str
    verification: VerificationResult | None
    evidence: list[EvidenceSpan]
    explanation: str
    citation_valid: bool
    retries_used: int
    citation_precision: float = 0.0
    unsupported_sentence_rate: float = 0.0


@dataclass
class ReflectionLoop:
    retriever: object
    verifier: object
    ranker: object | None = None
    confidence_threshold: float = 0.6
    max_retries: int = 2
    explanation_generator: object | None = None
    _retrieval_passes: int = field(default=0, init=False)

    def run(self, claim: str, *, top_k: int = 5) -> ReflectionOutcome:
        evidence = retrieve_evidence(self.retriever, claim, top_k=top_k).evidence
        evidence = rank_evidence(self.ranker, claim, evidence)

        retries = 0
        while retries <= self.max_retries:
            verification = verify_claim(self.verifier, claim, evidence)
            if verification.confidence < self.confidence_threshold:
                return ReflectionOutcome(
                    decision="abstain",
                    verification=verification,
                    evidence=evidence,
                    explanation=verification.explanation,
                    citation_valid=False,
                    retries_used=retries,
                )

            context = build_grounded_context(claim, evidence, top_k=top_k)
            explanation_output = explain_claim(
                context,
                verification,
                generator=self.explanation_generator,
            )
            citation_result = validate_citations(explanation_output.explanation, context)
            if citation_result.valid:
                return ReflectionOutcome(
                    decision="final",
                    verification=verification,
                    evidence=evidence,
                    explanation=explanation_output.explanation,
                    citation_valid=True,
                    retries_used=retries,
                    citation_precision=citation_result.citation_precision,
                    unsupported_sentence_rate=citation_result.unsupported_sentence_rate,
                )

            retries += 1
            if retries > self.max_retries:
                return ReflectionOutcome(
                    decision="abstain",
                    verification=verification,
                    evidence=evidence,
                    explanation=explanation_output.explanation,
                    citation_valid=False,
                    retries_used=self.max_retries,
                    citation_precision=citation_result.citation_precision,
                    unsupported_sentence_rate=citation_result.unsupported_sentence_rate,
                )

        return ReflectionOutcome(
            decision="abstain",
            verification=None,
            evidence=evidence,
            explanation="",
            citation_valid=False,
            retries_used=self.max_retries,
        )
