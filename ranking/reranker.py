"""Optional cross-encoder reranking helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from data.schemas import EvidenceSpan


@dataclass
class CrossEncoderReranker:  # pragma: no cover - optional dependency
    """Score claim/evidence pairs with a sentence-transformers CrossEncoder."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16
    model: object | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            from sentence_transformers import CrossEncoder  # type: ignore

            self.model = CrossEncoder(self.model_name)
        self.backend_name = "cross-encoder"

    def score_pairs(self, claim: str, evidence_list: Sequence[EvidenceSpan | str]) -> list[float]:
        texts = [span.text if isinstance(span, EvidenceSpan) else str(span) for span in evidence_list]
        pairs = [(claim, text) for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size)  # type: ignore[union-attr]
        return [float(score) for score in scores]

    def rerank(self, claim: str, evidence_list: Sequence[EvidenceSpan]) -> list[EvidenceSpan]:
        if not evidence_list:
            return []
        scores = self.score_pairs(claim, evidence_list)
        reranked = sorted(
            zip(evidence_list, scores, strict=False),
            key=lambda item: (-item[1], item[0].doc_id),
        )
        return [
            EvidenceSpan(
                doc_id=span.doc_id,
                text=span.text,
                title=span.title,
                score=float(score),
                metadata=dict(span.metadata),
            )
            for span, score in reranked
        ]

