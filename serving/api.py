"""FastAPI application for the Veritas demo."""

from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException

from agent.reflection import ReflectionOutcome
from models.deberta_verifier import VerificationResult
from rag import ContextBundle, build_context
from serving.cache import ResponseCache
from serving.model_loader import VerificationPipeline, load_pipeline
from serving.monitoring import MetricsTracker, elapsed_ms, measure_latency
from serving.schemas import EvidenceItem, VerifyRequest, VerifyResponse

app = FastAPI(title="Veritas", version="0.1.0")

_pipeline: VerificationPipeline = load_pipeline(
    evidence_corpus_path=os.getenv("VERITAS_EVIDENCE_CORPUS"),
    verifier_checkpoint=os.getenv("VERITAS_VERIFIER_CHECKPOINT"),
)
_cache = ResponseCache(ttl_seconds=120)
_metrics = MetricsTracker()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "verifier_backend": _pipeline.verifier_backend,
        "fallback_used": _pipeline.fallback_used,
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest) -> VerifyResponse:
    if len(request.claim) > 512:
        raise HTTPException(status_code=400, detail="claim too long")

    cache_key = f"{request.claim}:{request.top_k}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    start = measure_latency()
    outcome = _pipeline.reflection_loop.run(request.claim, top_k=request.top_k)
    evidence = [
        EvidenceItem(
            doc_id=item.doc_id,
            text=item.text,
            title=item.title,
            score=item.score,
            citation_id=index + 1,
        )
        for index, item in enumerate(outcome.evidence)
    ]
    response = VerifyResponse(
        verdict=outcome.verification.verdict if outcome.verification else "NOT ENOUGH INFO",
        confidence=outcome.verification.confidence if outcome.verification else 0.0,
        explanation=outcome.explanation or (outcome.verification.explanation if outcome.verification else ""),
        citation_valid=outcome.citation_valid,
        evidence=evidence,
        fallback_used=_pipeline.fallback_used,
        latency_ms=elapsed_ms(start),
    )
    _metrics.record(response.verdict, response.confidence, response.fallback_used, response.latency_ms)
    _cache.set(cache_key, response)
    return response


@app.get("/metrics")
def metrics() -> dict[str, object]:
    payload = _metrics.snapshot()
    payload["cache_entries"] = _cache.size()
    payload["fallback_used_for_default_demo"] = _pipeline.fallback_used
    payload["verifier_backend"] = _pipeline.verifier_backend
    return payload
