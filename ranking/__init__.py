"""Ranking package for Veritas."""

from .ab_simulation import BootstrapSummary, bootstrap_ab_simulation
from .features import RankingFeatureConfig, extract_features, feature_names, to_matrix
from .learned_ranker import LearnedRanker
from .metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k

__all__ = [
    "BootstrapSummary",
    "LearnedRanker",
    "RankingFeatureConfig",
    "bootstrap_ab_simulation",
    "extract_features",
    "feature_names",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "to_matrix",
]
