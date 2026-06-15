from data.schemas import EvidenceSpan
from models import DebertaVerifier, MockVerifier, ModelRouter, VALID_LABELS, normalize_label


def test_normalize_label_accepts_common_variants() -> None:
    assert normalize_label("nei") == "NOT ENOUGH INFO"
    assert set(VALID_LABELS) == {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}


def test_mock_verifier_returns_valid_label() -> None:
    verifier = MockVerifier()
    result = verifier.predict("Paris is in France", [EvidenceSpan(doc_id="1", text="Paris is in France.")])

    assert result.verdict in VALID_LABELS
    assert 0.0 <= result.confidence <= 1.0


def test_model_router_falls_back_when_checkpoint_missing() -> None:
    router = ModelRouter(verifier_checkpoint="does-not-exist")
    result = router.predict("Paris is in France", [])

    assert result.verdict == "NOT ENOUGH INFO"
    assert result.model_name == "mock"


def test_mock_verifier_per_passage_is_invariant_to_evidence_order() -> None:
    verifier = MockVerifier(aggregation_mode="per_passage_max", support_threshold=0.5, refute_threshold=0.5)
    evidence = [
        EvidenceSpan(doc_id="1", text="This sentence is unrelated."),
        EvidenceSpan(doc_id="2", text="Paris is in France."),
    ]

    first = verifier.predict("Paris is in France", evidence)
    second = verifier.predict("Paris is in France", list(reversed(evidence)))

    assert first.verdict == "SUPPORTED"
    assert second.verdict == "SUPPORTED"
    assert first.confidence == second.confidence
