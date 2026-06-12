"""RAG and citation utilities for Veritas."""

from .citation_checker import CitationCheckResult, check_citations
from .context_builder import ContextBundle, ContextEvidence, build_context
from .explanation_generator import ExplanationOutput, generate_explanation, generate_template_explanation

__all__ = [
    "CitationCheckResult",
    "ContextBundle",
    "ContextEvidence",
    "ExplanationOutput",
    "build_context",
    "check_citations",
    "generate_explanation",
    "generate_template_explanation",
]
