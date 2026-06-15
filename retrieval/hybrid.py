"""Hybrid retrieval using reciprocal rank fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from data.schemas import EvidenceSpan
from retrieval.query_expansion import expand_query


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
    title_retriever: object | None = None
    rrf_k: int = 60
    bm25_top_k: int = 10
    dense_top_k: int = 10
    title_top_k: int = 0
    query_expansion_top_k: int = 0
    final_top_k: int | None = None

    def retrieve(self, query: str, top_k: int = 5) -> list[EvidenceSpan]:
        requested_top_k = self.final_top_k or top_k
        rankings: list[list[EvidenceSpan]] = []
        rankings.append(self.bm25_retriever.retrieve(query, top_k=max(self.bm25_top_k, requested_top_k)))
        rankings.append(self.dense_retriever.retrieve(query, top_k=max(self.dense_top_k, requested_top_k)))
        if self.title_retriever is not None and self.title_top_k > 0:
            rankings.append(self.title_retriever.retrieve(query, top_k=max(self.title_top_k, requested_top_k)))
        if self.query_expansion_top_k > 0:
            for expanded_query in expand_query(query):
                rankings.append(self.bm25_retriever.retrieve(expanded_query, top_k=self.query_expansion_top_k))
                if self.title_retriever is not None and self.title_top_k > 0:
                    rankings.append(self.title_retriever.retrieve(expanded_query, top_k=min(self.title_top_k, self.query_expansion_top_k)))

        scores = reciprocal_rank_fusion(
            [[span.doc_id for span in ranking] for ranking in rankings if ranking],
            k=self.rrf_k,
        )
        lookup: dict[str, EvidenceSpan] = {}
        for ranking in rankings:
            for span in ranking:
                current = lookup.get(span.doc_id)
                if current is None or (span.score or 0.0) >= (current.score or 0.0):
                    lookup[span.doc_id] = span
        fused: list[EvidenceSpan] = []
        for doc_id, score in scores[:requested_top_k]:
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
