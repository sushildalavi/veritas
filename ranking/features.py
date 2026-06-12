"""Feature extraction for learned evidence ranking."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from data.schemas import EvidenceSpan

_DATE_PATTERNS = [
    r"\b\d{4}\b",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
]
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True)
class RankingFeatureConfig:
    include_cross_encoder_hook: bool = True


def extract_features(
    claim: str,
    evidence: EvidenceSpan,
    *,
    bm25_score: float = 0.0,
    dense_score: float = 0.0,
    cross_encoder_score: float | None = None,
    bm25_rank: int | None = None,
    dense_rank: int | None = None,
) -> dict[str, float]:
    claim_tokens = _tokenize(claim)
    evidence_tokens = _tokenize(evidence.text)
    claim_numbers = set(_NUMBER_PATTERN.findall(claim))
    evidence_numbers = set(_NUMBER_PATTERN.findall(evidence.text))
    claim_dates = _extract_dates(claim)
    evidence_dates = _extract_dates(evidence.text)

    features = {
        "bm25_score": float(bm25_score),
        "dense_similarity_score": float(dense_score),
        "cross_encoder_score": float(cross_encoder_score if cross_encoder_score is not None else evidence.score or 0.0),
        "lexical_overlap": _jaccard(claim_tokens, evidence_tokens),
        "number_overlap": _jaccard(claim_numbers, evidence_numbers),
        "date_overlap": _jaccard(claim_dates, evidence_dates),
        "evidence_length": float(len(evidence_tokens)),
        "claim_length": float(len(claim_tokens)),
        "bm25_rank_position": float(bm25_rank if bm25_rank is not None else 0),
        "dense_rank_position": float(dense_rank if dense_rank is not None else 0),
    }
    return features


def feature_names() -> list[str]:
    return [
        "bm25_score",
        "dense_similarity_score",
        "cross_encoder_score",
        "lexical_overlap",
        "number_overlap",
        "date_overlap",
        "evidence_length",
        "claim_length",
        "bm25_rank_position",
        "dense_rank_position",
    ]


def to_matrix(feature_rows: list[dict[str, float]]) -> list[list[float]]:
    names = feature_names()
    return [[row.get(name, 0.0) for name in names] for row in feature_rows]


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _extract_dates(text: str) -> set[str]:
    hits: set[str] = set()
    lowered = text.lower()
    for pattern in _DATE_PATTERNS:
        hits.update(re.findall(pattern, lowered))
    return hits


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0
