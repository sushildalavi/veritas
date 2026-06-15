"""Shared evidence formatting utilities for verifier and retrieval workflows."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
import re
from typing import Iterable, Literal

from data.schemas import EvidenceSpan

EvidenceStyle = Literal["plain", "passage", "evidence_letter", "bullet", "bracket"]

_WHITESPACE_PATTERN = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b[a-z]{3,}\b")
_STOPWORDS = frozenset(
    "the and for are but not you all can her was one our out day get has him his how its now old see two who did did not "
    "does from into know let man new other over said she the their them then there they this was way with".split()
)
_MARKER_PATTERNS = (
    re.compile(r"^\[(?:E\d+|\d+)\]\s*", re.IGNORECASE),
    re.compile(r"^passage\s+\d+\s*:\s*", re.IGNORECASE),
    re.compile(r"^evidence\s+[a-z]\s*:\s*", re.IGNORECASE),
    re.compile(r"^[\-\*\u2022]\s+"),
)


@dataclass(frozen=True)
class EvidenceBlock:
    text: str
    title: str | None = None


def sanitize_evidence_text(text: str) -> str:
    return "\n".join(split_evidence_blocks(text))


def split_evidence_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for raw_line in text.splitlines():
        cleaned = _clean_block(raw_line)
        if cleaned:
            blocks.append(cleaned)
    if blocks:
        return blocks
    cleaned = _clean_block(text)
    return [cleaned] if cleaned else []


def canonicalize_evidence_blocks(blocks: Iterable[str]) -> list[str]:
    unique: dict[str, str] = {}
    for block in blocks:
        cleaned = _clean_block(block)
        if cleaned:
            unique.setdefault(cleaned.lower(), cleaned)
    return [unique[key] for key in sorted(unique)]


def format_verifier_text(claim: str, evidence: str | list[EvidenceSpan]) -> str:
    if isinstance(evidence, list):
        evidence_text = render_evidence(
            evidence,
            style="plain",
            include_title=False,
            canonicalize=True,
        )
    else:
        evidence_text = "\n".join(canonicalize_evidence_blocks(split_evidence_blocks(evidence)))
    return f"Claim: {claim}\nEvidence: {evidence_text}"


def render_evidence(
    passages: list[EvidenceSpan],
    *,
    style: EvidenceStyle = "plain",
    include_title: bool = True,
    canonicalize: bool = False,
    shuffle: bool = False,
    rng: Random | None = None,
) -> str:
    blocks = [
        EvidenceBlock(
            text=_clean_block(span.text),
            title=_clean_block(span.title or "") if include_title and span.title else None,
        )
        for span in passages
        if _clean_block(span.text)
    ]
    if canonicalize:
        blocks = _canonicalize_blocks(blocks)
    elif shuffle and len(blocks) > 1:
        shuffled = list(blocks)
        (rng or Random()).shuffle(shuffled)
        blocks = shuffled
    return "\n".join(_format_block(block, index=index, style=style) for index, block in enumerate(blocks, start=1))


def choose_evidence_style(seed_text: str) -> EvidenceStyle:
    styles: tuple[EvidenceStyle, ...] = ("plain", "passage", "evidence_letter", "bullet", "bracket")
    return styles[sum(ord(char) for char in seed_text) % len(styles)]


def _canonicalize_blocks(blocks: list[EvidenceBlock]) -> list[EvidenceBlock]:
    unique: dict[tuple[str, str | None], EvidenceBlock] = {}
    for block in blocks:
        key = (block.text.lower(), block.title.lower() if block.title else None)
        unique.setdefault(key, block)
    return [unique[key] for key in sorted(unique, key=lambda item: (item[1] or "", item[0]))]


def _format_block(block: EvidenceBlock, *, index: int, style: EvidenceStyle) -> str:
    title_prefix = f"{block.title}: " if block.title else ""
    body = f"{title_prefix}{block.text}"
    if style == "plain":
        return body
    if style == "passage":
        return f"Passage {index}: {body}"
    if style == "evidence_letter":
        letter = chr(ord("A") + index - 1)
        return f"Evidence {letter}: {body}"
    if style == "bullet":
        return f"- {body}"
    return f"[E{index}] {body}"


def lexical_token_overlap(claim: str, passage: str) -> float:
    """Fraction of non-trivial claim tokens that also appear in passage.

    Mirrors the lexical_overlap column in the retrieved-evidence dataset
    (built by ``scripts/build_retrieved_evidence_dataset.py``) so the same
    threshold value can be reused in the serving gate.
    """
    claim_tokens = {t for t in _WORD_RE.findall(claim.lower()) if t not in _STOPWORDS}
    if not claim_tokens:
        return 0.0
    passage_tokens = {t for t in _WORD_RE.findall(passage.lower()) if t not in _STOPWORDS}
    return len(claim_tokens & passage_tokens) / len(claim_tokens)


def _clean_block(text: str) -> str:
    cleaned = _WHITESPACE_PATTERN.sub(" ", text.strip())
    if not cleaned:
        return ""
    changed = True
    while changed:
        changed = False
        for pattern in _MARKER_PATTERNS:
            updated = pattern.sub("", cleaned).strip()
            if updated != cleaned:
                cleaned = updated
                changed = True
    return cleaned
