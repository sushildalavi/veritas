"""Grounded explanation generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
    evidence_text = lead.evidence.text.rstrip(".!?")
    if verification.verdict == "SUPPORTED":
        conclusion = "supports the claim"
    elif verification.verdict == "REFUTED":
        conclusion = "contradicts the claim"
    else:
        conclusion = "does not establish the claim"
    explanation = f"[{lead.citation_id}] {evidence_text} and therefore {conclusion}."
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
    payload = _parse_json_response(response)
    if payload is not None:
        explanation = str(payload.get("explanation", response))
        citations = _extract_citations(payload.get("citations"))
        return ExplanationOutput(explanation=explanation, citations=citations)

    citations = sorted(
        {
            int(token.strip("[]"))
            for token in response.split()
            if token.startswith("[") and token.endswith("]") and token.strip("[]").isdigit()
        }
    )
    return ExplanationOutput(explanation=response, citations=citations)


def _parse_json_response(response: str) -> dict[str, object] | None:
    cleaned = response.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _extract_citations(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    citations: list[int] = []
    for item in value:
        if isinstance(item, int):
            citations.append(item)
        elif isinstance(item, str) and item.strip().isdigit():
            citations.append(int(item.strip()))
    return sorted(set(citations))
