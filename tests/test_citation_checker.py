from data.schemas import EvidenceSpan
from models.deberta_verifier import VerificationResult
from rag import build_context, check_citations, generate_template_explanation


def test_context_builder_assigns_citations() -> None:
    context = build_context(
        "Paris is in France",
        [
            EvidenceSpan(doc_id="1", text="Paris is the capital of France."),
            EvidenceSpan(doc_id="2", text="France is in Europe."),
        ],
        top_k=2,
    )

    assert context.evidence_items[0].citation_id == 1
    assert "[1]" in context.format_block()


def test_template_explanation_includes_citation_and_is_validated() -> None:
    context = build_context("Paris is in France", [EvidenceSpan(doc_id="1", text="Paris is the capital of France.")])
    verification = VerificationResult(verdict="SUPPORTED", confidence=0.9, logits={}, model_name="mock")

    output = generate_template_explanation(context, verification)
    result = check_citations(output.explanation, context)

    assert output.citations == [1]
    assert result.citation_precision == 1.0
    assert result.valid is True


def test_sentence_splitter_handles_verdict_prefix() -> None:
    context = build_context("Paris is in France", [EvidenceSpan(doc_id="1", text="Paris is the capital of France.")])

    result = check_citations("Verdict: SUPPORTED based on [1]; the claim is grounded in that evidence [1]", context)

    assert result.valid is True
