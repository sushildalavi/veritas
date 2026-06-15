from core.evidence_formatting import (
    canonicalize_evidence_blocks,
    format_verifier_text,
    render_evidence,
    sanitize_evidence_text,
)
from data.schemas import EvidenceSpan


def test_sanitize_evidence_text_removes_marker_variants() -> None:
    raw = "[E1] Alpha fact\nPassage 2: Beta fact\nEvidence C: Gamma fact\n- Delta fact"

    cleaned = sanitize_evidence_text(raw)

    assert cleaned == "Alpha fact\nBeta fact\nGamma fact\nDelta fact"


def test_format_verifier_text_is_invariant_to_marker_style() -> None:
    bracketed = "[E1] Alpha fact\n[E2] Beta fact"
    passage = "Passage 1: Alpha fact\nPassage 2: Beta fact"
    bulleted = "- Alpha fact\n- Beta fact"

    assert format_verifier_text("claim", bracketed) == format_verifier_text("claim", passage)
    assert format_verifier_text("claim", passage) == format_verifier_text("claim", bulleted)


def test_format_verifier_text_canonicalizes_multi_passage_order() -> None:
    first = "Passage 1: Zebra evidence\nPassage 2: Alpha evidence"
    second = "Evidence A: Alpha evidence\nEvidence B: Zebra evidence"

    assert format_verifier_text("claim", first) == format_verifier_text("claim", second)


def test_render_evidence_can_randomize_style_without_touching_content() -> None:
    passages = [
        EvidenceSpan(doc_id="1", text="Alpha fact"),
        EvidenceSpan(doc_id="2", text="Beta fact"),
    ]

    rendered = render_evidence(passages, style="evidence_letter", include_title=False)

    assert "Evidence A: Alpha fact" in rendered
    assert "Evidence B: Beta fact" in rendered
    assert canonicalize_evidence_blocks(rendered.splitlines()) == ["Alpha fact", "Beta fact"]
