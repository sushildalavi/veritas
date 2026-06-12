"""Local vector store with optional FAISS acceleration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import warnings


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if not denominator:
        return 0.0
    return _dot(left, right) / denominator


@dataclass
class LocalVectorStore:
    """A dependency-light vector store with an optional FAISS backend."""

    dimension: int

    def __post_init__(self) -> None:
        self.vectors: list[list[float]] = []
        self.metadata: list[dict[str, object]] = []
        self._faiss = None
        try:  # pragma: no cover - optional dependency
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                import faiss  # type: ignore

            self._faiss = faiss.IndexFlatIP(self.dimension)
        except Exception:
            self._faiss = None

    def add(self, vectors: Sequence[Sequence[float]], metadata: Sequence[dict[str, object]] | None = None) -> None:
        items = [list(vector) for vector in vectors]
        self.vectors.extend(items)
        self.metadata.extend(list(metadata) if metadata is not None else [{} for _ in items])
        if self._faiss is not None:  # pragma: no cover - optional dependency
            import numpy as np

            self._faiss.add(np.asarray(items, dtype="float32"))

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[dict[str, object]]:
        if not self.vectors:
            return []
        if self._faiss is not None:  # pragma: no cover - optional dependency
            import numpy as np

            scores, indices = self._faiss.search(np.asarray([query_vector], dtype="float32"), top_k)
            return self._pack_results(indices[0].tolist(), scores[0].tolist())

        scored = [
            (index, _cosine(query_vector, vector))
            for index, vector in enumerate(self.vectors)
        ]
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]
        return self._pack_results([index for index, _ in ranked], [score for _, score in ranked])

    def _pack_results(self, indices: list[int], scores: list[float]) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for index, score in zip(indices, scores):
            if index < 0:
                continue
            results.append(
                {
                    "index": index,
                    "score": float(score),
                    "metadata": self.metadata[index],
                }
            )
        return results
