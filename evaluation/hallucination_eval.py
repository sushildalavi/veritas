"""Hallucination and unsupported sentence evaluation."""

from __future__ import annotations

from rag.citation_checker import CitationCheckResult


def hallucination_rate(check_result: CitationCheckResult) -> float:
    return check_result.unsupported_sentence_rate
