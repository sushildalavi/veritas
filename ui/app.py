"""Gradio UI for the Veritas demo."""

from __future__ import annotations

import gradio as gr

from serving.api import verify
from serving.schemas import VerifyRequest


THEME_CSS = """
.veritas-shell {max-width: 1200px; margin: 0 auto;}
.veritas-hero {padding: 1rem 0 1.5rem 0;}
.veritas-title {font-size: 3rem; line-height: 1; font-weight: 800; letter-spacing: -0.04em;}
.veritas-subtitle {margin-top: 0.5rem; color: #4b5563; font-size: 1.05rem;}
.veritas-grid {display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 1rem;}
.veritas-card {border: 1px solid rgba(17, 24, 39, 0.08); border-radius: 18px; background: white; padding: 1rem 1.1rem; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);}
.veritas-metadata {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.75rem;}
.veritas-pill {display: inline-block; padding: 0.3rem 0.6rem; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 0.8rem; font-weight: 700;}
.veritas-muted {color: #6b7280;}
.veritas-section-title {font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem;}
.veritas-divider {height: 1px; background: rgba(17, 24, 39, 0.08); margin: 0.9rem 0;}
.gradio-container {background: linear-gradient(180deg, #f8fafc 0%, #ffffff 40%, #f8fafc 100%);}
"""


def _format_evidence_table(response) -> list[list[object]]:
    return [
        [item.citation_id, item.doc_id, item.title or "", item.score if item.score is not None else "", item.text]
        for item in response.evidence
    ]


def _example_claims() -> list[tuple[str, str]]:
    return [
        ("SUPPORTED", "Paris is the capital of France."),
        ("REFUTED", "The Earth is flat."),
        ("NOT ENOUGH INFO", "Veritas was founded in 2024."),
    ]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Veritas") as demo:
        with gr.Column(elem_classes=["veritas-shell"]):
            gr.HTML(f"<style>{THEME_CSS}</style>")
            gr.Markdown(
                """
                <div class="veritas-hero">
                  <div class="veritas-pill">Mac-only research demo</div>
                  <div class="veritas-title">Veritas - Evidence-Grounded Fact Verification</div>
                  <div class="veritas-subtitle">Hybrid retrieval, transformer verification, and citation-grounded explanations.</div>
                </div>
                """,
            )

            with gr.Row():
                with gr.Column(scale=2):
                    claim = gr.Textbox(label="Claim", placeholder="Enter a factual claim", lines=3)
                    top_k = gr.Slider(1, 10, value=5, step=1, label="Top K Evidence")
                    mode = gr.Radio(["Fast Mode", "Research Mode"], value="Fast Mode", label="Mode")
                    explanation_mode = gr.Radio(["Template", "MLX LoRA", "Preference Reranked"], value="Template", label="Explanation mode")
                    verify_button = gr.Button("Verify", variant="primary")
                    gr.Markdown("### Examples")
                    examples = gr.Examples(
                        examples=[[text, 5] for _, text in _example_claims()],
                        inputs=[claim, top_k],
                        label=None,
                    )
                with gr.Column(scale=2):
                    verdict = gr.Textbox(label="Verdict")
                    confidence = gr.Number(label="Confidence")
                    citation_valid = gr.Checkbox(label="Citation valid", interactive=False)
                    fallback_used = gr.Checkbox(label="Fallback used", interactive=False)
                    latency_ms = gr.Number(label="Latency (ms)")
                    request_id = gr.Textbox(label="Request ID")

            with gr.Tabs():
                with gr.Tab("Result"):
                    with gr.Row():
                        with gr.Column():
                            explanation = gr.Textbox(label="Explanation", lines=6)
                            evidence_table = gr.Dataframe(
                                headers=["Citation", "Doc ID", "Title", "Score", "Text"],
                                datatype=["number", "str", "str", "number", "str"],
                                label="Evidence",
                                interactive=False,
                            )
                        with gr.Column():
                            backend_used = gr.Textbox(label="Verifier backend")
                            model_name = gr.Textbox(label="Model")
                            explanation_mode_out = gr.Textbox(label="Explanation mode")
                            retrieval_backend = gr.Textbox(label="Retrieval backend")
                            reranker_backend = gr.Textbox(label="Reranker backend")
                            retrieval_fallback_used = gr.Checkbox(label="Retrieval fallback", interactive=False)
                            reranker_fallback_used = gr.Checkbox(label="Reranker fallback", interactive=False)

                with gr.Tab("Architecture"):
                    gr.Markdown(
                        """
                        <div class="veritas-card">
                        <div class="veritas-section-title">Pipeline</div>
                        Claim -> data checks -> BM25 -> dense retrieval -> RRF -> cross-encoder reranking -> DistilRoBERTa / DeBERTa verdict -> Qwen MLX LoRA explanation -> preference reranking -> citation checks -> API / UI
                        </div>
                        """,
                    )

                with gr.Tab("Limitations"):
                    gr.Markdown(
                        """
                        <div class="veritas-card">
                        <div class="veritas-section-title">What is not claimed</div>
                        <ul>
                          <li>No CUDA QLoRA claim.</li>
                          <li>No CUDA DPO claim.</li>
                          <li>No SOTA claim.</li>
                          <li>No production-scale benchmark claim.</li>
                        </ul>
                        </div>
                        """,
                    )

        def submit(claim_text: str, top_k_value: int, selected_mode: str, selected_explanation_mode: str):
            response = verify(VerifyRequest(claim=claim_text, top_k=int(top_k_value)))
            return (
                response.verdict,
                response.confidence,
                response.citation_valid,
                response.fallback_used,
                response.latency_ms,
                response.request_id,
                response.explanation,
                _format_evidence_table(response),
                response.backend_used,
                response.model_name,
                response.explanation_mode,
                response.retrieval_backend,
                response.reranker_backend,
                response.retrieval_fallback_used,
                response.reranker_fallback_used,
            )

        verify_button.click(
            submit,
            inputs=[claim, top_k, mode, explanation_mode],
            outputs=[
                verdict,
                confidence,
                citation_valid,
                fallback_used,
                latency_ms,
                request_id,
                explanation,
                evidence_table,
                backend_used,
                model_name,
                explanation_mode_out,
                retrieval_backend,
                reranker_backend,
                retrieval_fallback_used,
                reranker_fallback_used,
            ],
        )
    return demo


if __name__ == "__main__":  # pragma: no cover - UI entrypoint
    build_demo().launch()
