"""FastAPI application for the Veritas workspace."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import load_project_settings
from serving.cache import ResponseCache
from serving.errors import (
    artifact_missing,
    backend_unavailable,
    claim_too_long,
    report_not_found,
)
from serving.model_loader import VerificationPipeline, load_pipeline
from serving.monitoring import MetricsTracker, elapsed_ms, measure_latency
from serving.schemas import (
    EvidenceItem,
    ExplainRequest,
    ExplainResponse,
    LatencyBreakdown,
    MetadataResponse,
    MetricsSummaryResponse,
    PipelineRequest,
    PipelineResponse,
    RetrieveRequest,
    RetrieveResponse,
    VerifyRequest,
    VerifyResponse,
)
from serving.startup import check_artifacts, log_startup_report

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

# -- CORS origins -----------------------------------------------------------
_CORS_ORIGINS = os.environ.get(
    "VERITAS_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:8080",
).split(",")

# -- Allowlisted report files -----------------------------------------------
_ALLOWED_REPORTS: dict[str, Path] = {
    "final-results": ROOT / "reports" / "final_veritas_metrics_summary.md",
    "training-artifacts": ROOT / "docs" / "training_artifacts.md",
    "error-analysis": ROOT / "reports" / "error_analysis_650.md",
    "retrieval-ceiling": ROOT / "docs" / "retrieval_ceiling.md",
    "inference-performance": ROOT / "docs" / "inference_performance.md",
}

# -- App & Middleware --------------------------------------------------------
app = FastAPI(
    title="Veritas",
    version="1.0.0",
    description="AI-powered fact verification — submit a claim, get an evidence-backed verdict.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# -- Singletons -------------------------------------------------------------
_settings = load_project_settings()
_pipeline: VerificationPipeline = load_pipeline(settings=_settings)
_cache = ResponseCache(ttl_seconds=_settings.cache_ttl_seconds)
_metrics = MetricsTracker()
_artifact_checks: dict = {}


@app.on_event("startup")
def _startup() -> None:
    global _artifact_checks
    _artifact_checks = check_artifacts()
    log_startup_report(_artifact_checks)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", summary="Service liveness and pipeline metadata")
def health() -> dict:
    return {
        "status": "ok",
        "max_claim_length": _settings.max_claim_length,
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


# ---------------------------------------------------------------------------
# GET /metadata
# ---------------------------------------------------------------------------

@app.get("/metadata", response_model=MetadataResponse, summary="Project info, measured metrics, artifact availability")
def metadata() -> MetadataResponse:
    available_backends: list[str] = [_pipeline.verifier_backend]
    if _pipeline.explanation_generator is not None:
        available_backends.append(_pipeline.explanation_mode)

    return MetadataResponse(
        project="Veritas",
        version="1.0.0",
        description=(
            "Claim review pipeline with retrieval, NLI verification, citation checks, "
            "and measured validation artifacts. The app surfaces uncertainty instead of "
            "overstating weak evidence."
        ),
        verifier_checkpoint=str(_pipeline.checkpoint_path or "mock"),
        verifier_model=_pipeline.model_name,
        verifier_oracle_macro_f1=0.6728,
        verifier_retrieved_macro_f1=0.3887,
        retrieval_recall_at_10=0.5334,
        oracle_retrieved_gap=0.2841,
        retrieval_profile=_pipeline.retrieval_backend,
        available_backends=available_backends,
        endpoints=[
            "GET /health",
            "GET /metadata",
            "GET /metrics/summary",
            "POST /verify",
            "POST /retrieve",
            "POST /explain",
            "POST /pipeline",
            "GET /reports/{name}",
        ],
        artifact_checks=_artifact_checks,
    )


# ---------------------------------------------------------------------------
# GET /metrics/summary
# ---------------------------------------------------------------------------

@app.get("/metrics/summary", response_model=MetricsSummaryResponse, summary="Runtime stats snapshot (requests, latency, cache)")
def metrics_summary() -> MetricsSummaryResponse:
    snap = _metrics.snapshot()
    return MetricsSummaryResponse(
        requests=snap["requests"],
        fallbacks=snap["fallbacks"],
        citation_valids=_metrics.citation_valids,
        average_latency_ms=snap["average_latency_ms"],
        p95_latency_ms=snap["p95_latency_ms"],
        cache_entries=_cache.size(),
        backend_usage=dict(snap.get("backend_usage_counts", {})),
        verdicts=dict(snap.get("verdicts", {})),
    )


@app.get("/metrics")
def metrics() -> dict:
    payload = _metrics.snapshot()
    payload["cache_entries"] = _cache.size()
    payload["max_claim_length"] = _settings.max_claim_length
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


# ---------------------------------------------------------------------------
# POST /verify
# ---------------------------------------------------------------------------

@app.post("/verify", response_model=VerifyResponse, summary="BM25 retrieval + NLI verification with response cache")
def verify(request: VerifyRequest) -> VerifyResponse:
    if len(request.claim) > _settings.max_claim_length:
        raise claim_too_long(_settings.max_claim_length)

    start = measure_latency()
    request_id = str(uuid4())
    cache_key = f"verify:{request.claim}:{request.top_k}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached.model_copy(update={"request_id": request_id, "latency_ms": elapsed_ms(start)})

    try:
        outcome = _pipeline.reflection_loop.run(request.claim, top_k=request.top_k)
    except Exception as exc:
        LOGGER.error("verification failed: %s", exc)
        raise artifact_missing(
            "verifier",
            "Run: python3 scripts/train_transformer_verifier_clean.py",
        ) from exc

    response = _build_verify_response(outcome, request_id=request_id, latency_ms=elapsed_ms(start))
    _record_metrics(response)
    _cache.set(cache_key, response)
    return response


# ---------------------------------------------------------------------------
# POST /retrieve
# ---------------------------------------------------------------------------

@app.post("/retrieve", response_model=RetrieveResponse, summary="BM25 evidence retrieval only (no verification)")
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    if len(request.claim) > _settings.max_claim_length:
        raise claim_too_long(_settings.max_claim_length)

    start = measure_latency()
    cache_key = f"retrieve:{request.claim}:{request.top_k}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached.model_copy(update={"latency_ms": elapsed_ms(start)})

    try:
        from agent.tools import retrieve_evidence
        result = retrieve_evidence(_pipeline.retriever, request.claim, top_k=request.top_k)
        spans = result.evidence
    except Exception as exc:
        LOGGER.error("retrieval failed: %s", exc)
        raise artifact_missing(
            "retriever",
            "Check that evidence corpus is available at data/demo_corpus.jsonl",
        ) from exc

    items = [
        EvidenceItem(
            doc_id=s.doc_id,
            text=s.text,
            title=s.title,
            score=s.score,
            citation_id=i + 1,
        )
        for i, s in enumerate(spans)
    ]
    response = RetrieveResponse(
        claim=request.claim,
        evidence=items,
        retrieval_backend=_pipeline.retrieval_backend,
        latency_ms=elapsed_ms(start),
    )
    _cache.set(cache_key, response)
    return response


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------

@app.post("/explain", response_model=ExplainResponse, summary="Citation-grounded explanation generation (template or MLX LoRA)")
def explain(request: ExplainRequest) -> ExplainResponse:
    from agent.tools import build_grounded_context, explain_claim
    from data.schemas import EvidenceSpan
    from models.deberta_verifier import VerificationResult

    start = measure_latency()

    spans = [
        EvidenceSpan(
            doc_id=e.doc_id,
            text=e.text,
            title=e.title or "",
            score=e.score or 0.0,
        )
        for e in request.evidence
    ]

    fake_verification = VerificationResult(
        verdict=request.label,
        confidence=1.0,
        explanation="",
    )

    try:
        context = build_grounded_context(request.claim, spans)
        output = explain_claim(context, fake_verification, generator=_pipeline.explanation_generator)
        backend = _pipeline.explanation_mode if _pipeline.explanation_generator is not None else "template"
    except Exception as exc:
        LOGGER.warning("explanation generation failed: %s", exc)
        output_explanation = f"Claim is {request.label.lower()} based on provided evidence."
        return ExplainResponse(
            explanation=output_explanation,
            citations=[f"E{i+1}" for i in range(min(len(request.evidence), 3))],
            backend_used="template",
            latency_ms=elapsed_ms(start),
        )

    return ExplainResponse(
        explanation=output.explanation,
        citations=[f"E{c}" for c in output.citations],
        backend_used=backend,
        latency_ms=elapsed_ms(start),
    )


# ---------------------------------------------------------------------------
# POST /pipeline
# ---------------------------------------------------------------------------

@app.post("/pipeline", response_model=PipelineResponse, summary="Full pipeline: retrieval → verification → explanation + latency breakdown")
def pipeline(request: PipelineRequest) -> PipelineResponse:
    if len(request.claim) > _settings.max_claim_length:
        raise claim_too_long(_settings.max_claim_length)

    total_start = measure_latency()
    request_id = str(uuid4())

    cache_key = f"pipeline:{request.claim}:{request.top_k}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached.model_copy(update={"request_id": request_id})

    try:
        t_retrieval = perf_counter()
        from agent.tools import retrieve_evidence, rank_evidence
        ret = retrieve_evidence(_pipeline.retriever, request.claim, top_k=request.top_k)
        evidence_spans = rank_evidence(_pipeline.reflection_loop.ranker, request.claim, ret.evidence)
        retrieval_ms = (perf_counter() - t_retrieval) * 1000.0

        t_verify = perf_counter()
        from agent.tools import verify_claim
        verification = verify_claim(_pipeline.reflection_loop.verifier, request.claim, evidence_spans)
        verification_ms = (perf_counter() - t_verify) * 1000.0

        t_explain = perf_counter()
        from agent.tools import build_grounded_context, explain_claim, validate_citations
        context = build_grounded_context(request.claim, evidence_spans)
        exp_output = explain_claim(context, verification, generator=_pipeline.explanation_generator)
        explanation_text = exp_output.explanation
        citation_list = [f"E{c}" for c in exp_output.citations]
        citation_check = validate_citations(explanation_text, context)
        citation_valid = citation_check.valid
        explanation_ms = (perf_counter() - t_explain) * 1000.0

    except Exception as exc:
        LOGGER.error("pipeline failed: %s", exc)
        raise artifact_missing(
            "pipeline",
            "Ensure verifier checkpoint and evidence corpus are available.",
        ) from exc

    items = [
        EvidenceItem(
            doc_id=s.doc_id,
            text=s.text,
            title=s.title,
            score=s.score,
            citation_id=i + 1,
        )
        for i, s in enumerate(evidence_spans)
    ]

    response = PipelineResponse(
        request_id=request_id,
        claim=request.claim,
        verdict=verification.verdict,
        confidence=verification.confidence,
        evidence=items,
        explanation=explanation_text,
        citations=citation_list,
        citation_valid=citation_valid,
        backend_used=_pipeline.verifier_backend,
        retrieval_backend=_pipeline.retrieval_backend,
        explanation_mode=_pipeline.explanation_mode,
        latency=LatencyBreakdown(
            retrieval_ms=retrieval_ms,
            verification_ms=verification_ms,
            explanation_ms=explanation_ms,
            total_ms=elapsed_ms(total_start),
        ),
    )
    _cache.set(cache_key, response)
    return response


# ---------------------------------------------------------------------------
# GET /reports/{name}
# ---------------------------------------------------------------------------

@app.get("/reports/{name}", summary="Serve allowlisted research report files (final-results, error-analysis, etc.)")
def get_report(name: str) -> dict:
    if name not in _ALLOWED_REPORTS:
        raise report_not_found(name)
    path = _ALLOWED_REPORTS[name]
    if not path.exists():
        raise report_not_found(name)
    return {"name": name, "content": path.read_text(encoding="utf-8"), "format": path.suffix.lstrip(".")}



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_verify_response(outcome, *, request_id: str, latency_ms: float) -> VerifyResponse:
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
    verification = outcome.verification
    return VerifyResponse(
        verdict=verification.verdict if verification else "NOT ENOUGH INFO",
        confidence=verification.confidence if verification else 0.0,
        explanation=outcome.explanation or (verification.explanation if verification else ""),
        citation_valid=outcome.citation_valid,
        backend_used=_pipeline.verifier_backend,
        retrieval_backend=_pipeline.retrieval_backend,
        retrieval_fallback_used=_pipeline.retrieval_fallback_used,
        reranker_backend=_pipeline.reranker_backend,
        reranker_fallback_used=_pipeline.reranker_fallback_used,
        evidence=evidence,
        fallback_used=_pipeline.fallback_used,
        latency_ms=latency_ms,
        model_name=_pipeline.model_name,
        verifier_macro_f1=_pipeline.verifier_macro_f1,
        request_id=request_id,
        explanation_mode=_pipeline.explanation_mode,
    )


def _record_metrics(response: VerifyResponse) -> None:
    _metrics.record(
        response.verdict,
        response.confidence,
        response.fallback_used,
        response.latency_ms,
        backend_used=response.backend_used,
        citation_valid=response.citation_valid,
    )
