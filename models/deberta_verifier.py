"""DeBERTa verifier wrapper with a deterministic lightweight fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import warnings
from typing import Any

import joblib
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from core.evidence_formatting import format_verifier_text, sanitize_evidence_text
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
                self.model_name = "sklearn-tfidf-logreg"
                return
            try:
                from transformers import pipeline  # type: ignore

                self._pipeline = pipeline("text-classification", model=str(self.checkpoint_path))
                self._backend = "transformers"
                self.model_name = _read_model_name(self.checkpoint_path) or self.model_name
                return
            except Exception as exc:
                LOGGER.warning("Falling back to mock verifier: %s", exc)
        self._pipeline = None
        self._backend = "mock"
        self.model_name = "mock"

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
    normalized = []
    for span in evidence:
        cleaned = sanitize_evidence_text(span.text)
        if not cleaned:
            continue
        normalized.append(
            EvidenceSpan(
                doc_id=span.doc_id,
                text=cleaned,
                title=span.title,
                score=span.score,
                metadata=dict(span.metadata),
            )
        )
    return format_verifier_text(claim, normalized)


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
    metadata = _read_checkpoint_metadata(checkpoint_path)
    expected_sklearn_version = metadata.get("sklearn_version") if metadata else None
    current_sklearn_version = sklearn.__version__
    suppress_version_warning = expected_sklearn_version and expected_sklearn_version != current_sklearn_version
    if suppress_version_warning:
        LOGGER.warning(
            "Verifier checkpoint sklearn version mismatch at %s: checkpoint=%s current=%s. "
            "Attempting load with compatibility warnings suppressed; fallback remains available.",
            candidate_path,
            expected_sklearn_version,
            current_sklearn_version,
        )
    try:
        if suppress_version_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                payload = joblib.load(candidate_path)
        else:
            payload = joblib.load(candidate_path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Could not load joblib verifier checkpoint: %s", exc)
        return None
    if isinstance(payload, dict) and "pipeline" in payload:
        return payload
    return {"pipeline": payload, "label_order": list(getattr(payload, "classes_", VALID_LABELS))}


def _read_model_name(checkpoint_path: Path) -> str | None:
    metadata = _read_checkpoint_metadata(checkpoint_path)
    model_name = metadata.get("model_name")
    if isinstance(model_name, str) and model_name:
        return model_name
    config_path = checkpoint_path / "config.json" if checkpoint_path.is_dir() else None
    if config_path and config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        name = config.get("_name_or_path") or config.get("model_type")
        return str(name) if name else None
    return None


def _read_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    metadata_path = checkpoint_path / "metadata.json" if checkpoint_path.is_dir() else checkpoint_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
