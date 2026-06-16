"""Pydantic schemas for the serving API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvidenceItem(BaseModel):
    doc_id: str
    text: str
    title: str | None = None
    score: float | None = None
    citation_id: int | None = None


# -- /verify ----------------------------------------------------------------

class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("claim")
    @classmethod
    def _strip_claim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim cannot be empty")
        return value


class VerifyResponse(BaseModel):
    request_id: str
    verdict: str
    confidence: float
    explanation: str
    citation_valid: bool
    explanation_mode: str
    backend_used: str
    retrieval_backend: str
    retrieval_fallback_used: bool
    reranker_backend: str
    reranker_fallback_used: bool
    evidence: list[EvidenceItem]
    fallback_used: bool
    latency_ms: float
    model_name: str
    verifier_macro_f1: float | None = None


# -- /retrieve ---------------------------------------------------------------

class RetrieveRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("claim")
    @classmethod
    def _strip_claim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim cannot be empty")
        return value


class RetrieveResponse(BaseModel):
    claim: str
    evidence: list[EvidenceItem]
    retrieval_backend: str
    latency_ms: float


# -- /explain ----------------------------------------------------------------

class ExplainRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=1000)
    label: str = Field(..., pattern="^(SUPPORTED|REFUTED|NOT ENOUGH INFO)$")
    evidence: list[EvidenceItem] = Field(..., min_length=1)

    @field_validator("claim")
    @classmethod
    def _strip_claim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim cannot be empty")
        return value


class ExplainResponse(BaseModel):
    explanation: str
    citations: list[str]
    backend_used: str
    latency_ms: float


# -- /pipeline ---------------------------------------------------------------

class PipelineRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("claim")
    @classmethod
    def _strip_claim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim cannot be empty")
        return value


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    verification_ms: float
    explanation_ms: float
    total_ms: float


class PipelineResponse(BaseModel):
    request_id: str
    claim: str
    verdict: str
    confidence: float
    evidence: list[EvidenceItem]
    explanation: str
    citations: list[str]
    citation_valid: bool
    backend_used: str
    retrieval_backend: str
    explanation_mode: str
    latency: LatencyBreakdown


# -- /metadata ---------------------------------------------------------------

class MetadataResponse(BaseModel):
    project: str
    version: str
    description: str
    verifier_checkpoint: str
    verifier_model: str
    verifier_oracle_macro_f1: float
    verifier_retrieved_macro_f1: float
    retrieval_recall_at_10: float
    oracle_retrieved_gap: float
    retrieval_profile: str
    available_backends: list[str]
    endpoints: list[str]
    artifact_checks: dict[str, Any]


# -- /metrics/summary --------------------------------------------------------

class MetricsSummaryResponse(BaseModel):
    requests: int
    fallbacks: int
    citation_valids: int
    average_latency_ms: float
    p95_latency_ms: float
    cache_entries: int
    backend_usage: dict[str, int]
    verdicts: dict[str, int]
