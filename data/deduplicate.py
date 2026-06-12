"""Deduplication helpers for claims and evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TypeVar

from .schemas import ClaimEvidenceRecord, EvidenceSpan

T = TypeVar("T", ClaimEvidenceRecord, EvidenceSpan)


def exact_deduplicate(items: Iterable[T], *, key: str = "text") -> list[T]:
    """Remove exact duplicates while preserving order."""

    seen: set[tuple[str, str | None]] = set()
    deduped: list[T] = []

    for item in items:
        value = _value_for_key(item, key)
        identity = (key, value)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(item)
    return deduped


def deduplicate_records(records: Iterable[ClaimEvidenceRecord]) -> list[ClaimEvidenceRecord]:
    """Remove duplicate claims by claim text and duplicate evidence spans by text."""

    deduped: list[ClaimEvidenceRecord] = []
    seen_claims: set[str] = set()

    for record in records:
        if record.claim in seen_claims:
            continue
        seen_claims.add(record.claim)
        evidence = tuple(exact_deduplicate(record.evidence, key="text"))
        deduped.append(replace(record, evidence=evidence))

    return deduped


def _value_for_key(item: object, key: str) -> str | None:
    value = getattr(item, key, None)
    if value is None:
        return None
    return str(value)


class NearDuplicateDetector:
    """Optional approximate deduplication hook.

    The project can use a MinHash-backed implementation when the dependency is
    installed. In minimal environments, the detector falls back to a simple
    token-Jaccard threshold so the interface remains available without a heavy
    runtime dependency.
    """

    def __init__(self, threshold: float = 0.9) -> None:
        self.threshold = threshold

    def is_near_duplicate(self, left: str, right: str) -> bool:
        left_tokens = _token_set(left)
        right_tokens = _token_set(right)
        if not left_tokens and not right_tokens:
            return True
        union = left_tokens | right_tokens
        if not union:
            return True
        return len(left_tokens & right_tokens) / len(union) >= self.threshold


def _token_set(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}
