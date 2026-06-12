"""FastAPI application for the Veritas demo."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from core.config import load_project_settings
from serving.cache import ResponseCache
from serving.model_loader import VerificationPipeline, load_pipeline
from serving.monitoring import MetricsTracker, elapsed_ms, measure_latency
from serving.schemas import EvidenceItem, VerifyRequest, VerifyResponse

app = FastAPI(title="Veritas", version="0.1.0")

_settings = load_project_settings()
_pipeline: VerificationPipeline = load_pipeline(
    settings=_settings,
)
_cache = ResponseCache(ttl_seconds=_settings.cache_ttl_seconds)
_metrics = MetricsTracker()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "verifier_backend": _pipeline.verifier_backend,
        "fallback_used": _pipeline.fallback_used,
        "retrieval_backend": _pipeline.retrieval_backend,
        "embedding_model": _pipeline.embedding_model,
        "retrieval_fallback_used": _pipeline.retrieval_fallback_used,
        "reranker_backend": _pipeline.reranker_backend,
        "cross_encoder_model": _pipeline.cross_encoder_model,
        "reranker_fallback_used": _pipeline.reranker_fallback_used,
        "checkpoint_path": _pipeline.checkpoint_path,
        "model_name": _pipeline.model_name,
        "verifier_macro_f1": _pipeline.verifier_macro_f1,
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest) -> VerifyResponse:
    if len(request.claim) > 512:
        raise HTTPException(status_code=400, detail="claim too long")

    start = measure_latency()
    cache_key = f"{request.claim}:{request.top_k}"
    cached = _cache.get(cache_key)
    if cached is not None:
        _metrics.record(
            cached.verdict,
            cached.confidence,
            cached.fallback_used,
            elapsed_ms(start),
            backend_used=cached.backend_used,
            citation_valid=cached.citation_valid,
        )
        return cached

    try:
        outcome = _pipeline.reflection_loop.run(request.claim, top_k=request.top_k)
    except Exception as exc:  # pragma: no cover - defensive serving guard
        raise HTTPException(status_code=503, detail="verification unavailable") from exc
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
        backend_used=_pipeline.verifier_backend,
        retrieval_backend=_pipeline.retrieval_backend,
        retrieval_fallback_used=_pipeline.retrieval_fallback_used,
        reranker_backend=_pipeline.reranker_backend,
        reranker_fallback_used=_pipeline.reranker_fallback_used,
        evidence=evidence,
        fallback_used=_pipeline.fallback_used,
        latency_ms=elapsed_ms(start),
        model_name=_pipeline.model_name,
        verifier_macro_f1=_pipeline.verifier_macro_f1,
    )
    _metrics.record(
        response.verdict,
        response.confidence,
        response.fallback_used,
        response.latency_ms,
        backend_used=response.backend_used,
        citation_valid=response.citation_valid,
    )
    _cache.set(cache_key, response)
    return response


@app.get("/metrics")
def metrics() -> dict[str, object]:
    payload = _metrics.snapshot()
    payload["cache_entries"] = _cache.size()
    payload["fallback_used_for_default_demo"] = _pipeline.fallback_used
    payload["verifier_backend"] = _pipeline.verifier_backend
    payload["retrieval_backend"] = _pipeline.retrieval_backend
    payload["embedding_model"] = _pipeline.embedding_model
    payload["retrieval_fallback_used"] = _pipeline.retrieval_fallback_used
    payload["reranker_backend"] = _pipeline.reranker_backend
    payload["cross_encoder_model"] = _pipeline.cross_encoder_model
    payload["reranker_fallback_used"] = _pipeline.reranker_fallback_used
    payload["checkpoint_path"] = _pipeline.checkpoint_path
    payload["model_name"] = _pipeline.model_name
    payload["verifier_macro_f1"] = _pipeline.verifier_macro_f1
    return payload
