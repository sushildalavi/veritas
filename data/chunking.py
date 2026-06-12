"""Configurable text chunking helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 256
    overlap: int = 32

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")


def chunk_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    config = config or ChunkConfig()
    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    step = config.chunk_size - config.overlap
    while start < len(tokens):
        end = min(start + config.chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
        start += step
    return chunks
