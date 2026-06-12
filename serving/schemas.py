"""Pydantic schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class EvidenceItem(BaseModel):
    doc_id: str
    text: str
    title: str | None = None
    score: float | None = None
    citation_id: int | None = None


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("claim")
    @classmethod
    def _strip_claim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim cannot be empty")
        return value


class VerifyResponse(BaseModel):
    verdict: str
    confidence: float
    explanation: str
    citation_valid: bool
    backend_used: str
    evidence: list[EvidenceItem]
    fallback_used: bool
    latency_ms: float
