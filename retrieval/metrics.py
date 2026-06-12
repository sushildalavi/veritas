"""Retrieval metric helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(rankings: Sequence[Sequence[str]], relevant_sets: Sequence[Iterable[str]]) -> float:
    scores: list[float] = []
    for ranking, relevant in zip(rankings, relevant_sets):
        relevant_set = set(relevant)
        score = 0.0
        for index, doc_id in enumerate(ranking, start=1):
            if doc_id in relevant_set:
                score = 1.0 / index
                break
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    gains = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = [1.0] * min(len(relevant_set), k)
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg else 0.0
