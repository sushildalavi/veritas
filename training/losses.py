"""Lightweight losses and label helpers."""

from __future__ import annotations

from models.labels import VALID_LABELS, normalize_label


def label_to_index(label: str) -> int:
    return VALID_LABELS.index(normalize_label(label))


def index_to_label(index: int) -> str:
    return VALID_LABELS[index]
