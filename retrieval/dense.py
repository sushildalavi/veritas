"""Dense retrieval interfaces with optional sentence-transformers support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Literal

from data.schemas import EvidenceSpan

DenseBackendName = Literal["auto", "hashing", "sentence-transformers"]


class EmbeddingBackend:
    backend_name = "base"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbedder(EmbeddingBackend):
    """Deterministic, dependency-light embedding backend for tests and CI."""

    backend_name = "hashing"

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_text(text) for text in texts]

    def _encode_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self.dimension):
                vector[index] += ((digest[index % len(digest)] / 255.0) - 0.5)
        return _normalize(vector)


class SentenceTransformerEmbedder(EmbeddingBackend):  # pragma: no cover - optional dependency
    backend_name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, row)) for row in embeddings]


def load_embedder(
    backend: DenseBackendName = "auto",
    model_name: str | None = None,
    *,
    hashing_dimension: int = 64,
    allow_fallback: bool = True,
) -> EmbeddingBackend:
    if backend == "hashing":
        return HashingEmbedder(dimension=hashing_dimension)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    try:  # pragma: no cover - optional dependency
        return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        if not allow_fallback:
            raise
        return HashingEmbedder(dimension=hashing_dimension)


def load_default_embedder(model_name: str | None = None) -> EmbeddingBackend:
    return load_embedder("auto", model_name)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


@dataclass
class DenseRetriever:
    passages: list[EvidenceSpan]
    embedder: EmbeddingBackend | None = None
    backend: DenseBackendName = "hashing"
    model_name: str | None = None
    hashing_dimension: int = 64

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = load_embedder(
                self.backend,
                self.model_name,
                hashing_dimension=self.hashing_dimension,
                allow_fallback=True,
            )
        self.backend_name = getattr(self.embedder, "backend_name", self.backend)
        self._passage_embeddings = self.embedder.encode([passage.text for passage in self.passages])

    def retrieve(self, query: str, top_k: int = 5) -> list[EvidenceSpan]:
        if not self.passages:
            return []
        query_embedding = self.embedder.encode([query])[0]
        scored = [
            (index, cosine_similarity(query_embedding, passage_embedding))
            for index, passage_embedding in enumerate(self._passage_embeddings)
        ]
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            EvidenceSpan(
                doc_id=self.passages[index].doc_id,
                text=self.passages[index].text,
                title=self.passages[index].title,
                score=float(score),
                metadata=dict(self.passages[index].metadata),
            )
            for index, score in ranked
        ]
