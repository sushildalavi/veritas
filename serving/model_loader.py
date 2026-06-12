"""Load the free-demo verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reflection import ReflectionLoop
from core.config import ProjectSettings, load_project_settings
from data.schemas import EvidenceSpan
from models.model_router import ModelRouter
from retrieval import BM25Retriever, build_passage_corpus

from data.demo_corpus import DEFAULT_DEMO_PASSAGES


@dataclass
class VerificationPipeline:
    retriever: BM25Retriever
    verifier: ModelRouter
    reflection_loop: ReflectionLoop
    fallback_used: bool = True
    verifier_backend: str = "mock"
    checkpoint_path: str | None = None


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
    retriever = BM25Retriever(passages)
    prefer_deberta = settings.verifier_backend.lower() != "mock"
    verifier = ModelRouter(verifier_checkpoint=verifier_checkpoint, prefer_deberta=prefer_deberta)
    reflection_loop = ReflectionLoop(retriever=retriever, verifier=verifier)
    backend = getattr(getattr(verifier, "_deberta", None), "_backend", "mock")
    fallback_used = backend == "mock"
    return VerificationPipeline(
        retriever=retriever,
        verifier=verifier,
        reflection_loop=reflection_loop,
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


def _resolve_checkpoint_path(settings: ProjectSettings) -> str | Path | None:
    if settings.legacy_verifier_checkpoint:
        return settings.legacy_verifier_checkpoint
    backend = settings.verifier_backend.lower()
    sklearn_path = Path(settings.sklearn_checkpoint)
    transformer_path = Path(settings.transformer_checkpoint)
    if backend == "mock":
        return None
    if backend == "transformer":
        if transformer_path.exists():
            return transformer_path
        if sklearn_path.exists():
            return sklearn_path
        return transformer_path
    if sklearn_path.exists():
        return sklearn_path
    if transformer_path.exists():
        return transformer_path
    return sklearn_path
