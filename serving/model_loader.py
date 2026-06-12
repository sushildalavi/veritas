"""Load the free-demo verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.reflection import ReflectionLoop
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


def load_pipeline(evidence_corpus_path: str | Path | None = None, verifier_checkpoint: str | Path | None = None) -> VerificationPipeline:
    passages = _load_passages(evidence_corpus_path)
    retriever = BM25Retriever(passages)
    verifier = ModelRouter(verifier_checkpoint=verifier_checkpoint)
    reflection_loop = ReflectionLoop(retriever=retriever, verifier=verifier)
    fallback_used = verifier_checkpoint is None or not Path(verifier_checkpoint).exists()
    return VerificationPipeline(
        retriever=retriever,
        verifier=verifier,
        reflection_loop=reflection_loop,
        fallback_used=fallback_used,
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
