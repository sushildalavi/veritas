"""Model routing for deployed CPU and offline paths."""

from __future__ import annotations

import logging
from pathlib import Path

from data.schemas import EvidenceSpan
from models.deberta_verifier import DebertaVerifier, MockVerifier, VerificationResult

LOGGER = logging.getLogger(__name__)


class ModelRouter:
    def __init__(self, verifier_checkpoint: str | Path | None = None, prefer_deberta: bool = True) -> None:
        self.verifier_checkpoint = Path(verifier_checkpoint) if verifier_checkpoint else None
        self.prefer_deberta = prefer_deberta
        self._mock = MockVerifier()
        self._deberta = DebertaVerifier(self.verifier_checkpoint) if self.prefer_deberta else None

    def predict(self, claim: str, evidence: list[EvidenceSpan]) -> VerificationResult:
        if self._deberta is not None and self._deberta._pipeline is not None:
            return self._deberta.predict(claim, evidence)
        LOGGER.warning("Using mock verifier fallback")
        return self._mock.predict(claim, evidence)

    @property
    def model_name(self) -> str:
        if self._deberta is not None and self._deberta._pipeline is not None:
            return self._deberta.model_name
        return self._mock.model_name
