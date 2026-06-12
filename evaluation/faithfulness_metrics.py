"""Faithfulness metrics."""

from __future__ import annotations

from rag.citation_checker import CitationCheckResult


def citation_precision(result: CitationCheckResult) -> float:
    return result.citation_precision


def unsupported_sentence_rate(result: CitationCheckResult) -> float:
    return result.unsupported_sentence_rate


def verdict_consistency(result: CitationCheckResult) -> bool:
    return result.verdict_consistency
