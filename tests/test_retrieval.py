from retrieval.bm25 import BM25Retriever, build_passage_corpus
from retrieval.dense import DenseRetriever, HashingEmbedder
from retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from retrieval.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from retrieval.vector_store import LocalVectorStore


def test_bm25_retriever_ranks_relevant_passage_first() -> None:
    passages = build_passage_corpus(
        [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "Ottawa is the capital of Canada.",
        ]
    )
    retriever = BM25Retriever(passages)

    results = retriever.retrieve("What is the capital of France?", top_k=2)

    assert results[0].doc_id == "0"
    assert results[0].score >= results[1].score


def test_hybrid_rrf_prefers_consistently_ranked_documents() -> None:
    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "a", "d"]],
        k=60,
    )

    assert fused[0][0] in {"a", "b"}
    assert {doc_id for doc_id, _ in fused[:2]} == {"a", "b"}


def test_retrieval_metrics_compute_expected_values() -> None:
    assert recall_at_k(["a", "b", "c"], {"c", "d"}, 3) == 0.5
    assert mean_reciprocal_rank([["b", "a"], ["x", "y"]], [{"a"}, {"z"}]) == 0.25
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) > 0.0


def test_dense_and_vector_store_use_deterministic_fallbacks() -> None:
    passages = build_passage_corpus(["red apple", "blue sky", "green grass"])
    dense = DenseRetriever(passages, embedder=HashingEmbedder(dimension=16))
    hybrid = HybridRetriever(BM25Retriever(passages), dense)

    dense_results = dense.retrieve("blue sky", top_k=1)
    hybrid_results = hybrid.retrieve("blue sky", top_k=1)

    assert dense_results[0].doc_id == "1"
    assert hybrid_results[0].doc_id in {"0", "1"}

    store = LocalVectorStore(dimension=16)
    embeddings = HashingEmbedder(dimension=16).encode(["red apple", "blue sky"])
    store.add(embeddings, metadata=[{"doc_id": "0"}, {"doc_id": "1"}])
    results = store.search(embeddings[1], top_k=1)

    assert results[0]["metadata"]["doc_id"] == "1"
