"""DeBERTa verifier wrapper with a deterministic lightweight fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import joblib

from data.schemas import EvidenceSpan
from models.labels import VALID_LABELS, normalize_label

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    verdict: str
    confidence: float
    explanation: str = ""
    logits: dict[str, float] = field(default_factory=dict)
    model_name: str = "mock"


class MockVerifier:
    """Deterministic verifier used for CI and free-demo fallback."""

    model_name = "mock"

    def predict(self, claim: str, evidence: list[EvidenceSpan]) -> VerificationResult:
        if not evidence:
            return VerificationResult(
                verdict="NOT ENOUGH INFO",
                confidence=0.34,
                explanation="No evidence was retrieved.",
                logits={"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 1.0},
                model_name=self.model_name,
            )

        claim_tokens = _tokens(claim)
        evidence_tokens = set().union(*(_tokens(span.text) for span in evidence))
        overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens or {"x"})
        verdict = "SUPPORTED" if overlap >= 0.45 else "NOT ENOUGH INFO"
        if any(_negates(span.text, claim) for span in evidence):
            verdict = "REFUTED"
        confidence = min(0.99, 0.45 + overlap / 2.0)
        explanation = _template_explanation(claim, evidence, verdict)
        logits = {
            "SUPPORTED": 0.7 if verdict == "SUPPORTED" else 0.1,
            "REFUTED": 0.7 if verdict == "REFUTED" else 0.1,
            "NOT ENOUGH INFO": 0.7 if verdict == "NOT ENOUGH INFO" else 0.1,
        }
        return VerificationResult(
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            logits=logits,
            model_name=self.model_name,
        )


class DebertaVerifier:
    """Optional DeBERTa verifier. Falls back to `MockVerifier` when unavailable."""

    def __init__(self, checkpoint_path: str | Path | None = None, model_name: str = "microsoft/deberta-v3-base") -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model_name = model_name
        self._fallback = MockVerifier()
        self._pipeline = None
        self._backend = "mock"
        self._label_order = list(VALID_LABELS)
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        if self.checkpoint_path and self.checkpoint_path.exists():
            loaded = _load_joblib_model(self.checkpoint_path)
            if loaded is not None:
                self._pipeline = loaded["pipeline"]
                self._label_order = loaded.get("label_order", list(getattr(self._pipeline, "classes_", VALID_LABELS)))
                self._backend = "sklearn"
                return
            try:
                from transformers import pipeline  # type: ignore

                self._pipeline = pipeline("text-classification", model=str(self.checkpoint_path))
                self._backend = "transformers"
                return
            except Exception as exc:
                LOGGER.warning("Falling back to mock verifier: %s", exc)
        self._pipeline = None
        self._backend = "mock"

    def predict(self, claim: str, evidence: list[EvidenceSpan]) -> VerificationResult:
        if self._pipeline is None:
            return self._fallback.predict(claim, evidence)

        if self._backend == "sklearn":
            return self._predict_with_sklearn(claim, evidence)

        text = _build_verification_input(claim, evidence)
        outputs = self._pipeline(text)
        if isinstance(outputs, list):
            outputs = outputs[0]
        label = normalize_label(outputs.get("label", "NOT ENOUGH INFO"))
        score = float(outputs.get("score", 0.0))
        return VerificationResult(
            verdict=label,
            confidence=score,
            explanation=_template_explanation(claim, evidence, label),
            logits={label: score},
            model_name=self.model_name,
        )

    def _predict_with_sklearn(self, claim: str, evidence: list[EvidenceSpan]) -> VerificationResult:
        text = _build_verification_input(claim, evidence)
        probabilities = self._pipeline.predict_proba([text])[0]
        classes = list(getattr(self._pipeline, "classes_", self._label_order))
        label_scores = {normalize_label(label): float(score) for label, score in zip(classes, probabilities)}
        verdict = max(label_scores, key=label_scores.get)
        confidence = float(label_scores[verdict])
        return VerificationResult(
            verdict=verdict,
            confidence=confidence,
            explanation=_template_explanation(claim, evidence, verdict),
            logits=label_scores,
            model_name=self.model_name,
        )


def _build_verification_input(claim: str, evidence: list[EvidenceSpan]) -> str:
    evidence_text = "\n".join(span.text for span in evidence)
    return f"claim: {claim}\nevidence: {evidence_text}"


def _template_explanation(claim: str, evidence: list[EvidenceSpan], verdict: str) -> str:
    evidence_summary = evidence[0].text if evidence else "No evidence retrieved."
    return f"Verdict: {verdict}. Claim: {claim}. Supporting evidence: {evidence_summary}"


def _tokens(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def _negates(evidence_text: str, claim: str) -> bool:
    claim_tokens = _tokens(claim)
    evidence_tokens = _tokens(evidence_text)
    return "not" in evidence_tokens and bool(claim_tokens & evidence_tokens)


def _load_joblib_model(checkpoint_path: Path) -> dict[str, Any] | None:
    candidate_path = checkpoint_path
    if checkpoint_path.is_dir():
        candidate_path = checkpoint_path / "model.joblib"
    if not candidate_path.exists():
        return None
    try:
        payload = joblib.load(candidate_path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Could not load joblib verifier checkpoint: %s", exc)
        return None
    if isinstance(payload, dict) and "pipeline" in payload:
        return payload
    return {"pipeline": payload, "label_order": list(getattr(payload, "classes_", VALID_LABELS))}
