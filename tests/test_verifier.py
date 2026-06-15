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


def test_deberta_pipeline_prediction_uses_truncation() -> None:
    verifier = DebertaVerifier.__new__(DebertaVerifier)
    seen_kwargs = {}

    def fake_pipeline(text, **kwargs):  # noqa: ANN001
        seen_kwargs.update(kwargs)
        return [
            {"label": "SUPPORTED", "score": 0.8},
            {"label": "REFUTED", "score": 0.1},
            {"label": "NOT ENOUGH INFO", "score": 0.1},
        ]

    verifier._pipeline = fake_pipeline
    verifier._backend = "transformers"
    verifier._fallback = MockVerifier()
    verifier.model_name = "fake"

    result = verifier._predict_single("Paris is in France", [EvidenceSpan(doc_id="1", text="Paris is in France." * 100)])

    assert result.verdict == "SUPPORTED"
    assert result.logits["SUPPORTED"] == 0.8
    assert result.logits["REFUTED"] == 0.1
    assert result.logits["NOT ENOUGH INFO"] == 0.1
    assert seen_kwargs == {"truncation": True, "max_length": 512, "top_k": None}


def test_mock_verifier_scores_each_passage() -> None:
    verifier = MockVerifier()
    results = verifier.score_evidence_passages(
        "Paris is in France",
        [
            EvidenceSpan(doc_id="1", text="Paris is in France."),
            EvidenceSpan(doc_id="2", text="Ottawa is in Canada."),
        ],
    )

    assert len(results) == 2
    assert results[0].verdict in VALID_LABELS
    assert results[1].verdict in VALID_LABELS
