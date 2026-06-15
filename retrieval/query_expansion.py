"""Lightweight query expansion helpers for retrieval."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_ENTITY_PATTERN = re.compile(r"(?:[A-Z][a-z0-9'\-]+(?:\s+|$)){1,4}")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def expand_query(query: str, *, max_expansions: int = 4) -> list[str]:
    cleaned = " ".join(query.split())
    expansions: list[str] = []
    seen = {cleaned.lower()}

    for match in _ENTITY_PATTERN.findall(cleaned):
        entity = " ".join(match.split())
        if entity and entity.lower() not in seen:
            expansions.append(entity)
            seen.add(entity.lower())
        if len(expansions) >= max_expansions:
            return expansions

    keywords = [
        token
        for token in _TOKEN_PATTERN.findall(cleaned)
        if token.lower() not in _STOPWORDS and len(token) > 2
    ]
    if keywords:
        keyword_query = " ".join(keywords[: min(6, len(keywords))])
        if keyword_query.lower() not in seen:
            expansions.append(keyword_query)
            seen.add(keyword_query.lower())
    if len(keywords) > 1:
        tail_query = " ".join(keywords[-min(4, len(keywords)) :])
        if tail_query.lower() not in seen:
            expansions.append(tail_query)

    return expansions[:max_expansions]
