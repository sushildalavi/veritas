from pathlib import Path
import sys
import types

from data.schemas import ClaimEvidenceRecord, EvidenceSpan
from retrieval.bm25 import BM25Retriever, build_passage_corpus
from retrieval.dense import DenseRetriever, HashingEmbedder, load_embedder
from retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from retrieval.indexing import build_index_text
from retrieval.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from retrieval.query_expansion import expand_query
from scripts.run_retrieval_eval import evaluate_retrieval, parse_top_k
from retrieval.vector_store import LocalVectorStore
from core.config import ProjectSettings
from serving.model_loader import _load_retrieval_runtime


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


def test_index_text_can_include_title_and_metadata_window() -> None:
    span = EvidenceSpan(
        doc_id="1",
        text="Body text",
        title="Paper title",
        metadata={"section": "Abstract", "next_sentence": "Neighbor text"},
    )

    indexed = build_index_text(span, include_title=True, include_metadata_window=True)

    assert indexed == "Paper title Body text Abstract Neighbor text"


def test_query_expansion_extracts_entities_and_keywords() -> None:
    expansions = expand_query("Barack Obama was born in Honolulu Hawaii")

    assert expansions
    assert any("Barack Obama" in expansion for expansion in expansions)


def test_dense_loader_supports_hashing_and_lazy_sentence_transformers(monkeypatch) -> None:
    hashing = load_embedder("hashing", hashing_dimension=8, allow_fallback=False)
    assert isinstance(hashing, HashingEmbedder)
    assert hashing.backend_name == "hashing"

    fake_module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=True):  # noqa: ANN001
            return [[float(len(text))] for text in texts]

    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embedder = load_embedder("sentence-transformers", model_name="fake-model", allow_fallback=False)
    assert embedder.backend_name == "sentence-transformers"
    assert getattr(embedder, "model_name") == "fake-model"


def test_live_retrieval_runtime_bm25_only_path() -> None:
    passages = build_passage_corpus(["red apple", "blue sky"])
    settings = ProjectSettings(retrieval_backend="bm25_only", use_neural_retrieval=False)

    runtime = _load_retrieval_runtime(passages, settings)

    assert runtime.retrieval_backend == "bm25_only"
    assert runtime.embedding_model is None
    assert runtime.fallback_used is False
    assert isinstance(runtime.retriever, BM25Retriever)


def test_live_retrieval_runtime_hashing_hybrid_path() -> None:
    passages = build_passage_corpus(["red apple", "blue sky"])
    settings = ProjectSettings(
        retrieval_backend="bm25_hashing_hybrid",
        use_neural_retrieval=False,
        bm25_top_k=25,
        dense_top_k=25,
        title_top_k=10,
        query_expansion_top_k=5,
        include_title_in_index=True,
    )

    runtime = _load_retrieval_runtime(passages, settings)

    assert runtime.retrieval_backend == "bm25_hashing_hybrid"
    assert runtime.embedding_model == "hashing"
    assert runtime.fallback_used is False
    assert isinstance(runtime.retriever, HybridRetriever)
    assert runtime.retriever.bm25_top_k == 25
    assert runtime.retriever.title_top_k == 10
    assert runtime.retriever.query_expansion_top_k == 5


def test_live_retrieval_runtime_dense_only_hashing_path() -> None:
    passages = build_passage_corpus(["red apple", "blue sky"])
    settings = ProjectSettings(
        retrieval_backend="dense_only",
        use_neural_retrieval=False,
        embedding_backend="hashing",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    runtime = _load_retrieval_runtime(passages, settings)

    assert runtime.retrieval_backend == "dense_only"
    assert runtime.embedding_model == "hashing"
    assert runtime.fallback_used is False
    assert isinstance(runtime.retriever, DenseRetriever)


def test_live_retrieval_runtime_sentence_transformer_hybrid_path(monkeypatch) -> None:
    passages = build_passage_corpus(["red apple", "blue sky"])
    settings = ProjectSettings(
        retrieval_backend="bm25_sentence_transformer_hybrid",
        use_neural_retrieval=True,
        embedding_model="fake-model",
    )

    fake_module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=True):  # noqa: ANN001
            return [[float(len(text))] for text in texts]

    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    runtime = _load_retrieval_runtime(passages, settings)

    assert runtime.retrieval_backend == "bm25_sentence_transformer_hybrid"
    assert runtime.embedding_model == "fake-model"
    assert runtime.fallback_used is False
    assert isinstance(runtime.retriever, HybridRetriever)


def test_live_retrieval_runtime_falls_back_when_sentence_transformer_loader_fails(monkeypatch) -> None:
    passages = build_passage_corpus(["red apple", "blue sky"])
    settings = ProjectSettings(
        retrieval_backend="bm25_sentence_transformer_hybrid",
        use_neural_retrieval=True,
        embedding_model="broken-model",
    )

    fake_module = types.ModuleType("sentence_transformers")

    class BrokenSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            raise RuntimeError(f"failed to load {model_name}")

    fake_module.SentenceTransformer = BrokenSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    runtime = _load_retrieval_runtime(passages, settings)

    assert runtime.retrieval_backend == "bm25_only"
    assert runtime.embedding_model is None
    assert runtime.fallback_used is True
    assert isinstance(runtime.retriever, BM25Retriever)


def test_retrieval_eval_report_schema_uses_requested_backend(monkeypatch) -> None:
    corpus = [
        EvidenceSpan(doc_id="0", text="Paris is the capital of France."),
        EvidenceSpan(doc_id="1", text="Ottawa is the capital of Canada."),
    ]
    record = ClaimEvidenceRecord(
        claim_id="c1",
        claim="Paris is the capital of France.",
        label="SUPPORTED",
        evidence=(corpus[0],),
        split="val",
        metadata={},
    )

    monkeypatch.setattr(
        "scripts.run_retrieval_eval.load_records",
        lambda data_dir: {"fever_val": [record], "scifact_val": []},
    )
    monkeypatch.setattr("scripts.run_retrieval_eval.load_evidence_corpus", lambda data_dir: corpus)

    report = evaluate_retrieval(
        data_dir=Path("data/processed"),
        split="val",
        max_queries=1,
        dense_backend="hashing",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=parse_top_k("1,5,10"),
        bm25_top_k=20,
        dense_top_k=20,
        title_top_k=5,
        query_expansion_top_k=5,
        rrf_k=60,
        include_title_in_index=True,
        include_metadata_window=False,
    )

    assert report["split"] == "val"
    assert report["dense_backend"] == "hashing"
    assert report["evidence_corpus_size"] == 2
    assert set(report["metrics"]) == {"bm25", "dense", "hybrid"}
    assert "recall@10" in report["metrics"]["hybrid"]
    assert "ndcg@5" in report["metrics"]["hybrid"]
    assert report["retrieval_config"]["title_top_k"] == 5
