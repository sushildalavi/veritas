from dataclasses import dataclass

from data.schemas import EvidenceSpan
from agent.reflection import ReflectionLoop


@dataclass
class StubRetriever:
    evidence: list[EvidenceSpan]

    def retrieve(self, claim: str, top_k: int = 5):
        return self.evidence[:top_k]


@dataclass
class StubVerifier:
    verdict: str = "SUPPORTED"
    confidence: float = 0.9

    def predict(self, claim: str, evidence: list[EvidenceSpan]):
        from models.deberta_verifier import VerificationResult

        return VerificationResult(
            verdict=self.verdict,
            confidence=self.confidence,
            explanation="stub",
            logits={self.verdict: self.confidence},
            model_name="stub",
        )


def test_reflection_loop_success_path() -> None:
    loop = ReflectionLoop(
        retriever=StubRetriever([EvidenceSpan(doc_id="1", text="Paris is the capital of France.")]),
        verifier=StubVerifier(),
        confidence_threshold=0.5,
    )

    outcome = loop.run("Paris is in France")

    assert outcome.decision == "final"
    assert outcome.citation_valid is True


def test_reflection_loop_abstains_on_low_confidence() -> None:
    loop = ReflectionLoop(
        retriever=StubRetriever([EvidenceSpan(doc_id="1", text="Paris is the capital of France.")]),
        verifier=StubVerifier(confidence=0.2),
        confidence_threshold=0.5,
    )

    outcome = loop.run("Paris is in France")

    assert outcome.decision == "abstain"
    assert outcome.verification.confidence == 0.2


def test_reflection_loop_regenerates_invalid_citation() -> None:
    calls = {"count": 0}

    def generator(prompt: str) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            return "Verdict: SUPPORTED based on [99]; unsupported citation"
        return "Verdict: SUPPORTED based on [1]; the claim is grounded in that evidence [1] because Paris is the capital of France."

    loop = ReflectionLoop(
        retriever=StubRetriever([EvidenceSpan(doc_id="1", text="Paris is the capital of France.")]),
        verifier=StubVerifier(),
        confidence_threshold=0.5,
        explanation_generator=generator,
        max_retries=2,
    )

    outcome = loop.run("Paris is in France")

    assert outcome.decision == "final"
    assert calls["count"] >= 2
