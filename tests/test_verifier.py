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
