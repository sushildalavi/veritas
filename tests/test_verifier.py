from core.evidence_formatting import lexical_token_overlap
from data.schemas import EvidenceSpan
from models import DebertaVerifier, MockVerifier, ModelRouter, VALID_LABELS, normalize_label
from models.deberta_verifier import VerificationResult, _aggregate_results, _gate_results


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


def test_per_passage_aggregation_prefers_refute_when_threshold_crosses() -> None:
    result = _aggregate_results(
        "Paris is in France",
        [
            EvidenceSpan(doc_id="1", text="Unrelated."),
            EvidenceSpan(doc_id="2", text="Paris is not in France."),
        ],
        [
            VerificationResult(verdict="NOT ENOUGH INFO", confidence=0.6, logits={"SUPPORTED": 0.2, "REFUTED": 0.1, "NOT ENOUGH INFO": 0.6}),
            VerificationResult(verdict="REFUTED", confidence=0.7, logits={"SUPPORTED": 0.1, "REFUTED": 0.7, "NOT ENOUGH INFO": 0.2}),
        ],
        model_name="fake",
        support_threshold=0.55,
        refute_threshold=0.5,
    )

    assert result.verdict == "REFUTED"
    assert result.confidence == 0.7


def test_lexical_token_overlap_matching_claim() -> None:
    overlap = lexical_token_overlap("Paris is in France", "Paris is the capital of France.")
    assert overlap > 0.0


def test_lexical_token_overlap_irrelevant_passage() -> None:
    overlap = lexical_token_overlap("The Eiffel Tower is in Paris", "Ottawa is the capital of Canada.")
    assert overlap == 0.0


def test_gate_results_forces_nei_below_threshold() -> None:
    evidence = [
        EvidenceSpan(doc_id="1", text="completely unrelated text about something else"),
        EvidenceSpan(doc_id="2", text="Paris is the capital of France and sits on the Seine river"),
    ]
    results = [
        VerificationResult(verdict="SUPPORTED", confidence=0.8, logits={"SUPPORTED": 0.8, "REFUTED": 0.1, "NOT ENOUGH INFO": 0.1}),
        VerificationResult(verdict="SUPPORTED", confidence=0.9, logits={"SUPPORTED": 0.9, "REFUTED": 0.05, "NOT ENOUGH INFO": 0.05}),
    ]
    gated = _gate_results("Paris is in France", evidence, results, gate_threshold=0.5, model_name="test")

    assert gated[0].verdict == "NOT ENOUGH INFO", "irrelevant passage should be gated to NEI"
    assert gated[1].verdict == "SUPPORTED", "relevant passage should pass through unchanged"


def test_gate_disabled_passes_all_results_through() -> None:
    verifier = MockVerifier(aggregation_mode="per_passage_max", relevance_gate_threshold=None)
    evidence = [EvidenceSpan(doc_id="1", text="completely irrelevant passage about dogs")]
    result = verifier.predict("Paris is in France", evidence)
    assert result.verdict in VALID_LABELS


def test_gate_threshold_zero_passes_all_through() -> None:
    verifier = MockVerifier(aggregation_mode="per_passage_max", relevance_gate_threshold=0.0)
    evidence = [EvidenceSpan(doc_id="1", text="completely irrelevant passage")]
    result = verifier.predict("Paris is in France", [evidence[0], evidence[0]])
    assert result.verdict in VALID_LABELS


def test_gate_threshold_high_forces_unrelated_claim_to_nei() -> None:
    # claim tokens: {rhinoceros, conservation} (unique, not in evidence)
    verifier = MockVerifier(aggregation_mode="per_passage_max", relevance_gate_threshold=0.9)
    evidence = [
        EvidenceSpan(doc_id="1", text="Paris is a beautiful city located in France"),
        EvidenceSpan(doc_id="2", text="The Eiffel Tower stands in Paris France"),
    ]
    result = verifier.predict("rhinoceros conservation efforts", evidence)
    assert result.verdict == "NOT ENOUGH INFO", "high gate threshold should force irrelevant passages to NEI"


def test_model_router_passes_gate_threshold() -> None:
    router = ModelRouter(verifier_checkpoint="does-not-exist", relevance_gate_threshold=0.5)
    assert router._mock.relevance_gate_threshold == 0.5


def test_per_passage_aggregation_falls_back_to_nei_below_thresholds() -> None:
    result = _aggregate_results(
        "Paris is in France",
        [EvidenceSpan(doc_id="1", text="Unrelated.")],
        [
            VerificationResult(
                verdict="NOT ENOUGH INFO",
                confidence=0.4,
                logits={"SUPPORTED": 0.4, "REFUTED": 0.3, "NOT ENOUGH INFO": 0.3},
            )
        ],
        model_name="fake",
        support_threshold=0.55,
        refute_threshold=0.5,
    )

    assert result.verdict == "NOT ENOUGH INFO"
