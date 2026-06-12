"""Citation validation and faithfulness helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from .context_builder import ContextBundle


@dataclass(frozen=True)
class CitationCheckResult:
    citation_precision: float
    unsupported_sentence_rate: float
    verdict_consistency: bool
    valid: bool
    missing_citations: list[int]
    unsupported_sentences: list[str]


def check_citations(
    explanation: str,
    context: ContextBundle,
    *,
    nli_validator: Callable[[str, str], bool] | None = None,
) -> CitationCheckResult:
    citation_ids = _extract_citations(explanation)
    available = {item.citation_id: item.evidence.text for item in context.evidence_items}
    missing_citations = sorted(citation_ids - set(available))
    cited = citation_ids & set(available)

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", explanation) if sentence.strip()]
    unsupported_sentences: list[str] = []
    for sentence in sentences:
        cited_ids = _extract_citations(sentence)
        if not cited_ids:
            unsupported_sentences.append(sentence)
            continue
        if nli_validator is not None:
            if not any(nli_validator(sentence, available[citation_id]) for citation_id in cited_ids if citation_id in available):
                unsupported_sentences.append(sentence)
            continue
        if not any(_lexically_supported(sentence, available[citation_id]) for citation_id in cited_ids if citation_id in available):
            unsupported_sentences.append(sentence)

    total_citations = len(citation_ids)
    citation_precision = len(cited) / total_citations if total_citations else 1.0
    unsupported_sentence_rate = len(unsupported_sentences) / len(sentences) if sentences else 0.0
    verdict_consistency = citation_precision >= 0.5 and unsupported_sentence_rate <= 0.5
    return CitationCheckResult(
        citation_precision=citation_precision,
        unsupported_sentence_rate=unsupported_sentence_rate,
        verdict_consistency=verdict_consistency,
        valid=not missing_citations and unsupported_sentence_rate == 0.0,
        missing_citations=missing_citations,
        unsupported_sentences=unsupported_sentences,
    )


def _extract_citations(text: str) -> set[int]:
    return {int(match) for match in re.findall(r"\[(\d+)\]", text)}


def _lexically_supported(sentence: str, evidence_text: str) -> bool:
    sentence_tokens = _tokens(sentence)
    evidence_tokens = _tokens(evidence_text)
    if not sentence_tokens:
        return False
    return len(sentence_tokens & evidence_tokens) / len(sentence_tokens) >= 0.2


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}
