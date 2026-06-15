"""Shared indexing helpers for sparse and dense retrieval."""

from __future__ import annotations

from data.schemas import EvidenceSpan

_WINDOW_METADATA_KEYS = ("section", "context_window", "prev_sentence", "next_sentence", "window")


def build_index_text(
    span: EvidenceSpan,
    *,
    include_title: bool = False,
    include_metadata_window: bool = False,
) -> str:
    parts: list[str] = []
    if include_title and span.title:
        parts.append(span.title)
    parts.append(span.text)
    if include_metadata_window:
        for key in _WINDOW_METADATA_KEYS:
            value = span.metadata.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return " ".join(part.strip() for part in parts if part and part.strip())
