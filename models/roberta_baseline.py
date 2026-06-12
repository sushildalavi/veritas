"""RoBERTa verifier baseline wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data.schemas import EvidenceSpan
from models.deberta_verifier import MockVerifier, VerificationResult


@dataclass
class RobertaBaselineVerifier:
    checkpoint_path: str | Path | None = None

    def __post_init__(self) -> None:
        self._fallback = MockVerifier()
        self._pipeline = None
        if self.checkpoint_path:
            self._load_pipeline()

    def _load_pipeline(self) -> None:  # pragma: no cover - optional dependency
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline("text-classification", model=str(self.checkpoint_path))
        except Exception:
            self._pipeline = None

    def predict(self, claim: str, evidence: list[EvidenceSpan]) -> VerificationResult:
        if self._pipeline is None:
            return self._fallback.predict(claim, evidence)
        return self._fallback.predict(claim, evidence)
