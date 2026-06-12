"""Load the free-demo verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reflection import ReflectionLoop
from core.config import ProjectSettings, load_project_settings
from data.schemas import EvidenceSpan
from models.model_router import ModelRouter
from retrieval import BM25Retriever, DenseRetriever, HashingEmbedder, HybridRetriever, build_passage_corpus

from data.demo_corpus import DEFAULT_DEMO_PASSAGES

LOGGER = logging.getLogger(__name__)


@dataclass
class VerificationPipeline:
    retriever: object
    verifier: ModelRouter
    reflection_loop: ReflectionLoop
    retrieval_backend: str = "bm25_only"
    embedding_model: str | None = None
    retrieval_fallback_used: bool = False
    fallback_used: bool = True
    verifier_backend: str = "mock"
    checkpoint_path: str | None = None


@dataclass
class RetrievalRuntime:
    retriever: object
    retrieval_backend: str
    embedding_model: str | None
    fallback_used: bool = False


def load_pipeline(
    evidence_corpus_path: str | Path | None = None,
    verifier_checkpoint: str | Path | None = None,
    *,
    settings: ProjectSettings | None = None,
) -> VerificationPipeline:
    settings = settings or load_project_settings()
    if evidence_corpus_path is None:
        evidence_corpus_path = settings.evidence_corpus_path
    if verifier_checkpoint is None:
        verifier_checkpoint = _resolve_checkpoint_path(settings)
    passages = _load_passages(evidence_corpus_path)
    retrieval_runtime = _load_retrieval_runtime(passages, settings)
    retriever = retrieval_runtime.retriever
    prefer_deberta = settings.verifier_backend.lower() != "mock"
    verifier = ModelRouter(verifier_checkpoint=verifier_checkpoint, prefer_deberta=prefer_deberta)
    reflection_loop = ReflectionLoop(retriever=retriever, verifier=verifier)
    backend = getattr(getattr(verifier, "_deberta", None), "_backend", "mock")
    fallback_used = backend == "mock"
    return VerificationPipeline(
        retriever=retriever,
        verifier=verifier,
        reflection_loop=reflection_loop,
        retrieval_backend=retrieval_runtime.retrieval_backend,
        embedding_model=retrieval_runtime.embedding_model,
        retrieval_fallback_used=retrieval_runtime.fallback_used,
        fallback_used=fallback_used,
        verifier_backend=backend,
        checkpoint_path=str(verifier_checkpoint) if verifier_checkpoint else None,
    )


def _load_passages(evidence_corpus_path: str | Path | None) -> list[EvidenceSpan]:
    if evidence_corpus_path is None:
        return build_passage_corpus(DEFAULT_DEMO_PASSAGES)
    path = Path(evidence_corpus_path)
    if not path.exists():
        return build_passage_corpus(DEFAULT_DEMO_PASSAGES)
    passages: list[EvidenceSpan] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import json

        payload = json.loads(line)
        passages.append(
            EvidenceSpan(
                doc_id=str(payload.get("doc_id", len(passages))),
                text=str(payload.get("text", "")),
                title=payload.get("title"),
                score=payload.get("score"),
            )
        )
    return passages or build_passage_corpus(DEFAULT_DEMO_PASSAGES)


def _load_retrieval_runtime(passages: list[EvidenceSpan], settings: ProjectSettings) -> RetrievalRuntime:
    backend = settings.retrieval_backend.strip().lower()
    if backend == "bm25_only" or not passages:
        return RetrievalRuntime(
            retriever=BM25Retriever(passages),
            retrieval_backend="bm25_only",
            embedding_model=None,
            fallback_used=False,
        )

    if backend == "bm25_hashing_hybrid":
        dense_retriever = DenseRetriever(passages, embedder=HashingEmbedder())
        return RetrievalRuntime(
            retriever=HybridRetriever(BM25Retriever(passages), dense_retriever),
            retrieval_backend="bm25_hashing_hybrid",
            embedding_model="hashing",
            fallback_used=False,
        )

    if backend == "bm25_sentence_transformer_hybrid":
        if not settings.use_neural_retrieval:
            return RetrievalRuntime(
                retriever=BM25Retriever(passages),
                retrieval_backend="bm25_only",
                embedding_model=None,
                fallback_used=False,
            )
        try:
            dense_retriever = DenseRetriever(
                passages,
                backend="sentence-transformers",
                model_name=settings.embedding_model,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("Falling back to BM25 retrieval after neural loader failure: %s", exc)
            return RetrievalRuntime(
                retriever=BM25Retriever(passages),
                retrieval_backend="bm25_only",
                embedding_model=None,
                fallback_used=True,
            )
        return RetrievalRuntime(
            retriever=HybridRetriever(BM25Retriever(passages), dense_retriever),
            retrieval_backend="bm25_sentence_transformer_hybrid",
            embedding_model=settings.embedding_model,
            fallback_used=False,
        )

    LOGGER.warning("Unknown retrieval backend %s; defaulting to BM25-only", backend)
    return RetrievalRuntime(
        retriever=BM25Retriever(passages),
        retrieval_backend="bm25_only",
        embedding_model=None,
        fallback_used=True,
    )


def _resolve_checkpoint_path(settings: ProjectSettings) -> str | Path | None:
    backend = settings.verifier_backend.lower()
    sklearn_path = Path(settings.sklearn_checkpoint)
    transformer_path = Path(settings.transformer_checkpoint)
    legacy_path = Path(settings.legacy_verifier_checkpoint) if settings.legacy_verifier_checkpoint else None
    if backend == "mock":
        return None
    if _is_transformer_checkpoint(transformer_path):
        return transformer_path
    if legacy_path is not None and _is_transformer_checkpoint(legacy_path):
        return legacy_path
    if _is_sklearn_checkpoint(sklearn_path):
        return sklearn_path
    if backend == "transformer":
        if transformer_path.exists():
            return transformer_path
        if legacy_path is not None and legacy_path.exists():
            return legacy_path
        if sklearn_path.exists():
            return sklearn_path
        return transformer_path
    if legacy_path is not None and legacy_path.exists():
        return legacy_path
    if sklearn_path.exists():
        return sklearn_path
    if transformer_path.exists():
        return transformer_path
    return legacy_path if legacy_path is not None and legacy_path.exists() else None


def _is_transformer_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix in {".bin", ".safetensors"}
    candidate_files = {"config.json", "model.safetensors", "pytorch_model.bin"}
    return any((path / name).exists() for name in candidate_files)


def _is_sklearn_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix == ".joblib"
    return (path / "model.joblib").exists()
