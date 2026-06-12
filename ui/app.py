"""Gradio UI for the Veritas demo."""

from __future__ import annotations

import gradio as gr

from serving.api import verify
from serving.schemas import VerifyRequest


def _format_evidence_table(response) -> list[list[object]]:
    return [
        [item.citation_id, item.doc_id, item.title or "", item.score if item.score is not None else "", item.text]
        for item in response.evidence
    ]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Veritas") as demo:
        gr.Markdown("# Veritas\nClaim verification demo.")
        claim = gr.Textbox(label="Claim", placeholder="Enter a factual claim")
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top K Evidence")
        verify_button = gr.Button("Verify")
        verdict = gr.Textbox(label="Verdict")
        confidence = gr.Number(label="Confidence")
        citation_valid = gr.Checkbox(label="Citation valid", interactive=False)
        explanation = gr.Textbox(label="Explanation", lines=5)
        evidence_table = gr.Dataframe(
            headers=["Citation", "Doc ID", "Title", "Score", "Text"],
            datatype=["number", "str", "str", "number", "str"],
            label="Evidence",
            interactive=False,
        )

        def submit(claim_text: str, top_k_value: int):
            response = verify(VerifyRequest(claim=claim_text, top_k=int(top_k_value)))
            return (
                response.verdict,
                response.confidence,
                response.citation_valid,
                response.explanation,
                _format_evidence_table(response),
            )

        verify_button.click(
            submit,
            inputs=[claim, top_k],
            outputs=[verdict, confidence, citation_valid, explanation, evidence_table],
        )
    return demo


if __name__ == "__main__":  # pragma: no cover - UI entrypoint
    build_demo().launch()
