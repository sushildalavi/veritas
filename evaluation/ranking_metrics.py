"""Ranking metric wrappers."""

from __future__ import annotations

from ranking.metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k

__all__ = ["mean_average_precision", "mean_reciprocal_rank", "ndcg_at_k"]
