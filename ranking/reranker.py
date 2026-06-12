"""Optional cross-encoder reranking helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from math import fsum

from data.schemas import EvidenceSpan


@dataclass
class HeuristicReranker:
    """Deterministic fallback ranker that scores lexical overlap."""

    backend_name: str = "heuristic"

    def rank(self, claim: str, evidence_list: Sequence[EvidenceSpan]) -> list[EvidenceSpan]:
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

    def score_pairs(self, claim: str, evidence_list: Sequence[EvidenceSpan | str]) -> list[float]:
        claim_tokens = _tokenize(claim)
        claim_set = set(claim_tokens)
        scores: list[float] = []
        for item in evidence_list:
            text = item.text if isinstance(item, EvidenceSpan) else str(item)
            evidence_tokens = _tokenize(text)
            if not evidence_tokens:
                scores.append(0.0)
                continue
            overlap = len(claim_set.intersection(evidence_tokens))
            density = overlap / max(len(evidence_tokens), 1)
            coverage = overlap / max(len(claim_set), 1)
            exact_match = 1.0 if claim.strip().lower() in text.lower() else 0.0
            scores.append(float(fsum([coverage, density, exact_match])))
        return scores


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

    def rank(self, claim: str, evidence_list: Sequence[EvidenceSpan]) -> list[EvidenceSpan]:
        return self.rerank(claim, evidence_list)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
