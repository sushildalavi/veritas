"""Lightweight dataset loading helpers."""

from __future__ import annotations

from collections.abc import Iterable


def load_hf_dataset(name: str, *, split: str = "train"):
    """Load a Hugging Face dataset when the optional dependency is installed."""

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is not installed") from exc

    return load_dataset(name, split=split)


def iter_rows(dataset: Iterable[dict]) -> Iterable[dict]:
    for row in dataset:
        yield dict(row)
