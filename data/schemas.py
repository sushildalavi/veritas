"""Shared claim/evidence data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceSpan:
    """A single evidence passage associated with a claim."""

    doc_id: str
    text: str
    title: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimEvidenceRecord:
    """A fact-checking example with claim, label, and evidence."""

    claim_id: str
    claim: str
    label: str
    evidence: tuple[EvidenceSpan, ...] = ()
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
