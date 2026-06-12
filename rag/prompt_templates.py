"""Prompt templates for grounded explanation generation."""

from __future__ import annotations

EXPLANATION_PROMPT = """You are a fact-checking assistant.
Claim: {claim}
Verdict: {verdict}
Evidence:
{evidence_block}
Write a concise explanation grounded in the evidence and cite each sentence with bracketed evidence ids.
"""

REFLECTION_PROMPT = """Review the claim, verdict, and evidence for citation support.
If the explanation lacks support, regenerate with stronger grounding.
"""
