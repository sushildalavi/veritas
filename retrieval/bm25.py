"""BM25 retrieval helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
import re

from data.schemas import EvidenceSpan
from retrieval.indexing import build_index_text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _FallbackBM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        scores: list[float] = []
        n_docs = len(self.corpus)
        for doc_index, doc in enumerate(self.corpus):
            score = 0.0
            doc_terms: dict[str, int] = {}
            for token in doc:
                doc_terms[token] = doc_terms.get(token, 0) + 1
            for term in query_tokens:
                freq = doc_terms.get(term, 0)
                if freq == 0:
                    continue
                df = self.df.get(term, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_index] / (self.avgdl or 1.0))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def _build_index(corpus: list[list[str]]):
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return _FallbackBM25(corpus)
    return BM25Okapi(corpus)


@dataclass
class BM25Retriever:
    """A lightweight BM25 retriever over evidence passages."""

    passages: list[EvidenceSpan]
    include_title_in_index: bool = False
    include_metadata_window: bool = False

    def __post_init__(self) -> None:
        self._tokenized = [
            tokenize(
                build_index_text(
                    span,
                    include_title=self.include_title_in_index,
                    include_metadata_window=self.include_metadata_window,
                )
            )
            for span in self.passages
        ]
        self._index = _build_index(self._tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[EvidenceSpan]:
        if not self.passages:
            return []
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
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


def build_passage_corpus(texts: Iterable[str]) -> list[EvidenceSpan]:
    return [EvidenceSpan(doc_id=str(index), text=text) for index, text in enumerate(texts)]
