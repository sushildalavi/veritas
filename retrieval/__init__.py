"""Retrieval package for Veritas."""

from .bm25 import BM25Retriever, build_passage_corpus
from .dense import DenseRetriever, HashingEmbedder
from .hybrid import HybridRetriever, reciprocal_rank_fusion
from .metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from .vector_store import LocalVectorStore

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HashingEmbedder",
    "HybridRetriever",
    "LocalVectorStore",
    "build_passage_corpus",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_fusion",
]
