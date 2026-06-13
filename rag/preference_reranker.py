"""Preference-guided explanation reranking for Mac-local MLX outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from rag.citation_checker import check_citations
from rag.context_builder import ContextBundle


@dataclass(frozen=True)
class CandidateScore:
    index: int
    score: float
    valid_json: bool
    verdict_matches: bool
    citation_valid: bool
    unsupported_sentence_rate: float
    explanation_length: int
    payload: dict[str, Any] | None
    raw_text: str
    reasons: tuple[str, ...]


def parse_candidate(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
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


def score_candidate(
    text: str,
    *,
    verifier_verdict: str,
    context: ContextBundle,
    index: int,
    concise_token_threshold: int = 80,
) -> CandidateScore:
    payload = parse_candidate(text)
    reasons: list[str] = []
    score = 0.0
    verdict_matches = False
    citation_valid = False
    unsupported_sentence_rate = 1.0
    explanation = ""

    if payload is not None:
        score += 1.0
        reasons.append("valid_json")
        verdict = str(payload.get("verdict", "")).strip().upper().replace(" ", "_")
        verdict_matches = verdict == verifier_verdict.upper().replace(" ", "_")
        if verdict_matches:
            score += 1.0
            reasons.append("verdict_matches")
        explanation = str(payload.get("explanation", ""))
        citations = payload.get("citations", [])
        if isinstance(citations, list) and citations:
            explanation_for_check = explanation
            citation_text = " ".join(f"[{item}]" for item in citations if isinstance(item, str))
            citation_check = check_citations(f"{explanation_for_check} {citation_text}".strip(), context)
            citation_valid = citation_check.valid
            unsupported_sentence_rate = citation_check.unsupported_sentence_rate
            if citation_valid:
                score += 1.0
                reasons.append("citation_valid")
            else:
                score -= 1.0
                reasons.append("citation_invalid")
            if unsupported_sentence_rate <= 0.25:
                score += 1.0
                reasons.append("supported_sentences")
            else:
                score -= unsupported_sentence_rate
                reasons.append("unsupported_sentences")
        else:
            score -= 1.0
            reasons.append("missing_citations")
    else:
        score -= 1.0
        reasons.append("invalid_json")

    explanation_length = len(explanation.split())
    if explanation_length and explanation_length <= concise_token_threshold:
        score += 0.5
        reasons.append("concise")
    elif explanation_length > concise_token_threshold:
        score -= 0.25
        reasons.append("too_long")

    if not verdict_matches:
        score -= 0.5

    return CandidateScore(
        index=index,
        score=round(score, 4),
        valid_json=payload is not None,
        verdict_matches=verdict_matches,
        citation_valid=citation_valid,
        unsupported_sentence_rate=unsupported_sentence_rate,
        explanation_length=explanation_length,
        payload=payload,
        raw_text=text,
        reasons=tuple(reasons),
    )


def select_best_candidate(
    candidates: list[str],
    *,
    verifier_verdict: str,
    context: ContextBundle,
    concise_token_threshold: int = 80,
) -> tuple[str, list[CandidateScore]]:
    scored = [
        score_candidate(
            candidate,
            verifier_verdict=verifier_verdict,
            context=context,
            index=index,
            concise_token_threshold=concise_token_threshold,
        )
        for index, candidate in enumerate(candidates)
    ]
    best = max(scored, key=lambda item: (item.score, -item.index))
    return best.raw_text, scored
