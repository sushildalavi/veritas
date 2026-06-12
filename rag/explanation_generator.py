"""Grounded explanation generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from models.deberta_verifier import VerificationResult
from .context_builder import ContextBundle
from .prompt_templates import EXPLANATION_PROMPT


@dataclass(frozen=True)
class ExplanationOutput:
    explanation: str
    citations: list[int]


def generate_template_explanation(context: ContextBundle, verification: VerificationResult) -> ExplanationOutput:
    if not context.evidence_items:
        return ExplanationOutput(
            explanation=f"Verdict: {verification.verdict}. No evidence was retrieved, so the claim cannot be grounded.",
            citations=[],
        )

    lead = context.evidence_items[0]
    explanation = (
        f"Verdict: {verification.verdict} based on [{lead.citation_id}]; "
        f"the claim '{context.claim}' is grounded in that evidence [{lead.citation_id}] because {lead.evidence.text}"
    )
    if len(context.evidence_items) > 1:
        tail = context.evidence_items[1]
        explanation += f"; additional context [{tail.citation_id}] reinforces the same verdict [{tail.citation_id}]"
    return ExplanationOutput(
        explanation=explanation,
        citations=[item.citation_id for item in context.evidence_items[:2]],
    )


def generate_explanation(
    context: ContextBundle,
    verification: VerificationResult,
    generator: Callable[[str], str] | None = None,
) -> ExplanationOutput:
    if generator is None:
        return generate_template_explanation(context, verification)

    prompt = EXPLANATION_PROMPT.format(
        claim=context.claim,
        verdict=verification.verdict,
        evidence_block=context.format_block(),
    )
    response = generator(prompt)
    citations = sorted(
        {
            int(token.strip("[]"))
            for token in response.split()
            if token.startswith("[") and token.endswith("]") and token.strip("[]").isdigit()
        }
    )
    return ExplanationOutput(explanation=response, citations=citations)
