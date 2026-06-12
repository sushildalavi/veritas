"""Shared verifier labels."""

from __future__ import annotations

VALID_LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")


def normalize_label(label: str) -> str:
    normalized = label.strip().upper()
    if normalized in {"NEI", "NOT_ENOUGH_INFO", "NOT ENOUGH INFO"}:
        return "NOT ENOUGH INFO"
    if normalized not in VALID_LABELS:
        raise ValueError(f"invalid label: {label}")
    return normalized
