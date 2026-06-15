"""Hybrid retrieval using reciprocal rank fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from data.schemas import EvidenceSpan


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    order: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
            order.setdefault(doc_id, len(order))
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))


@dataclass
class HybridRetriever:
    bm25_retriever: object
    dense_retriever: object
    rrf_k: int = 60

    def retrieve(self, query: str, top_k: int = 5) -> list[EvidenceSpan]:
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k)
        scores = reciprocal_rank_fusion(
            [[span.doc_id for span in bm25_results], [span.doc_id for span in dense_results]],
            k=self.rrf_k,
        )
        lookup: dict[str, EvidenceSpan] = {}
        for span in [*bm25_results, *dense_results]:
            current = lookup.get(span.doc_id)
            if current is None or (span.score or 0.0) >= (current.score or 0.0):
                lookup[span.doc_id] = span
        fused: list[EvidenceSpan] = []
        for doc_id, score in scores[:top_k]:
            span = lookup[doc_id]
            fused.append(
                EvidenceSpan(
                    doc_id=span.doc_id,
                    text=span.text,
                    title=span.title,
                    score=float(score),
                    metadata=dict(span.metadata),
                )
            )
        return fused
