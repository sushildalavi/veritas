"""Polished Gradio UI for the Veritas demo."""

from __future__ import annotations

from html import escape

import gradio as gr

from serving.api import verify
from serving.schemas import VerifyRequest


THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #07111f;
  --bg-elevated: rgba(10, 17, 31, 0.72);
  --surface: rgba(15, 23, 42, 0.76);
  --surface-strong: rgba(10, 16, 29, 0.92);
  --line: rgba(148, 163, 184, 0.16);
  --text: #e5eefb;
  --muted: #9fb2cc;
  --muted-strong: #c6d6ea;
  --accent: #7dd3fc;
  --accent-2: #f472b6;
  --accent-3: #a78bfa;
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #fb7185;
  --shadow: 0 30px 80px rgba(2, 6, 23, 0.45);
}

.gradio-container {
  background:
    radial-gradient(circle at 20% 20%, rgba(125, 211, 252, 0.10), transparent 24%),
    radial-gradient(circle at 80% 10%, rgba(244, 114, 182, 0.12), transparent 22%),
    radial-gradient(circle at 55% 70%, rgba(167, 139, 250, 0.12), transparent 28%),
    linear-gradient(180deg, #060b14 0%, #0a1222 46%, #050812 100%);
  color: var(--text);
  font-family: 'Space Grotesk', system-ui, sans-serif;
}

body, .gradio-container, .gradio-container * {
  box-sizing: border-box;
}

.veritas-shell {
  position: relative;
  max-width: 1440px;
  margin: 0 auto;
  padding: 28px 20px 40px;
}

.veritas-shell::before,
.veritas-shell::after {
  content: "";
  position: fixed;
  inset: auto;
  width: 34rem;
  height: 34rem;
  border-radius: 50%;
  filter: blur(40px);
  opacity: 0.45;
  pointer-events: none;
  z-index: 0;
  animation: floatBlob 18s ease-in-out infinite;
}

.veritas-shell::before {
  top: -6rem;
  right: -8rem;
  background: radial-gradient(circle, rgba(125, 211, 252, 0.35), rgba(125, 211, 252, 0.02) 60%, transparent 70%);
}

.veritas-shell::after {
  bottom: -8rem;
  left: -10rem;
  background: radial-gradient(circle, rgba(244, 114, 182, 0.30), rgba(244, 114, 182, 0.03) 60%, transparent 70%);
  animation-delay: -6s;
}

.veritas-content {
  position: relative;
  z-index: 1;
}

.veritas-hero {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 24px;
  align-items: stretch;
  margin-bottom: 22px;
  animation: riseIn 900ms cubic-bezier(.2,.8,.2,1) both;
}

.veritas-hero-main,
.veritas-hero-side,
.veritas-panel,
.veritas-card {
  border: 1px solid var(--line);
  border-radius: 24px;
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(8, 13, 24, 0.88));
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}

.veritas-hero-main {
  padding: 28px;
  min-height: 320px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  overflow: hidden;
  position: relative;
}

.veritas-hero-main::after {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(115deg, transparent 0%, rgba(255,255,255,0.04) 38%, transparent 58%),
    radial-gradient(circle at 20% 0%, rgba(125, 211, 252, 0.12), transparent 26%);
  transform: translateX(-18%);
  animation: sheen 7.5s ease-in-out infinite;
  pointer-events: none;
}

.veritas-kicker {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  width: fit-content;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(125, 211, 252, 0.25);
  background: rgba(125, 211, 252, 0.08);
  color: var(--accent);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 18px;
}

.veritas-title {
  font-size: clamp(2.6rem, 4vw, 4.6rem);
  line-height: 0.95;
  letter-spacing: -0.05em;
  font-weight: 700;
  margin: 0 0 14px 0;
  max-width: 14ch;
}

.veritas-subtitle {
  font-size: 1.08rem;
  line-height: 1.65;
  color: var(--muted);
  max-width: 70ch;
}

.veritas-hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 24px;
}

.veritas-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(2, 6, 23, 0.45);
  color: var(--muted-strong);
  font-size: 0.87rem;
}

.veritas-badge strong {
  color: var(--text);
  font-weight: 700;
}

.veritas-hero-side {
  padding: 20px;
  display: grid;
  grid-template-rows: auto 1fr auto;
  gap: 16px;
}

.veritas-signal {
  padding: 16px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(167, 139, 250, 0.12));
  border: 1px solid rgba(125, 211, 252, 0.22);
}

.veritas-signal h3,
.veritas-section h3 {
  margin: 0 0 8px;
  font-size: 1rem;
  letter-spacing: 0.01em;
}

.veritas-signal p,
.veritas-section p,
.veritas-note {
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
  font-size: 0.95rem;
}

.veritas-rail {
  display: grid;
  gap: 10px;
}

.veritas-stat {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 10px;
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(2, 6, 23, 0.35);
  border: 1px solid rgba(148, 163, 184, 0.12);
}

.veritas-stat .label {
  color: var(--muted);
  font-size: 0.84rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.veritas-stat .value {
  color: var(--text);
  font-size: 1.02rem;
  font-weight: 700;
  text-align: right;
}

.veritas-grid {
  display: grid;
  grid-template-columns: minmax(0, 0.98fr) minmax(0, 1.02fr);
  gap: 18px;
  align-items: start;
}

.veritas-panel {
  padding: 20px;
}

.veritas-panel-title {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 14px;
}

.veritas-panel-title h2 {
  margin: 0;
  font-size: 1.15rem;
}

.veritas-panel-title span {
  color: var(--muted);
  font-size: 0.88rem;
}

.veritas-hr {
  height: 1px;
  margin: 18px 0;
  background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.18), transparent);
}

.veritas-section {
  display: grid;
  gap: 14px;
}

.veritas-section .gradio-row,
.veritas-section .gradio-column {
  gap: 12px;
}

.veritas-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.veritas-chip {
  padding: 8px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(2, 6, 23, 0.34);
  color: var(--muted-strong);
  font-size: 0.84rem;
}

.veritas-chip.good { border-color: rgba(34, 197, 94, 0.24); color: #bbf7d0; }
.veritas-chip.warn { border-color: rgba(245, 158, 11, 0.24); color: #fde68a; }
.veritas-chip.info { border-color: rgba(125, 211, 252, 0.24); color: #d7f7ff; }

.veritas-card {
  padding: 16px 18px;
}

.veritas-card h4 {
  margin: 0 0 8px;
  font-size: 0.98rem;
}

.veritas-card p,
.veritas-card li {
  color: var(--muted);
  line-height: 1.55;
}

.veritas-card ul {
  margin: 10px 0 0 18px;
}

.veritas-result-hero {
  padding: 18px;
  border-radius: 20px;
  border: 1px solid rgba(148, 163, 184, 0.14);
  background:
    linear-gradient(180deg, rgba(2, 6, 23, 0.38), rgba(2, 6, 23, 0.22)),
    radial-gradient(circle at 20% 20%, rgba(125, 211, 252, 0.18), transparent 30%);
  margin-bottom: 14px;
}

.veritas-result-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.veritas-metric {
  padding: 14px;
  border-radius: 16px;
  background: rgba(2, 6, 23, 0.35);
  border: 1px solid rgba(148, 163, 184, 0.12);
}

.veritas-metric .metric-label {
  color: var(--muted);
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

.veritas-metric .metric-value {
  color: var(--text);
  font-size: 1.16rem;
  font-weight: 700;
}

.veritas-stack {
  display: grid;
  gap: 12px;
}

.veritas-footer {
  margin-top: 18px;
  color: var(--muted);
  font-size: 0.9rem;
}

.gr-button-primary {
  background: linear-gradient(135deg, #38bdf8 0%, #818cf8 55%, #f472b6 100%) !important;
  color: white !important;
  border: 0 !important;
  box-shadow: 0 12px 32px rgba(56, 189, 248, 0.25);
}

.gr-textbox, .gr-dropdown, .gr-slider, .gr-radio, .gr-dataframe, .gr-markdown, .gr-html {
  color: var(--text);
}

.gr-textbox textarea,
.gr-textbox input,
.gr-dropdown input,
.gr-slider input {
  background: rgba(2, 6, 23, 0.55) !important;
  color: var(--text) !important;
}

@keyframes riseIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes floatBlob {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(12px, -16px, 0) scale(1.04); }
}

@keyframes sheen {
  0%, 100% { transform: translateX(-18%); opacity: 0.45; }
  50% { transform: translateX(14%); opacity: 0.85; }
}

@media (max-width: 1100px) {
  .veritas-hero,
  .veritas-grid,
  .veritas-result-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .veritas-shell {
    padding: 18px 12px 24px;
  }

  .veritas-hero-main {
    min-height: auto;
    padding: 22px;
  }

  .veritas-title {
    max-width: none;
  }
}
"""


EXAMPLES = [
    "Paris is the capital of France.",
    "The Earth is flat.",
    "Veritas was founded in 2024.",
    "Scream 2 is a slasher film.",
]


def _format_value(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "n/a"
    return str(value)


def _example_claims() -> list[list[object]]:
    return [[claim, 5] for claim in EXAMPLES]


def _build_chip(label: str, value: object, tone: str = "info") -> str:
    return f'<span class="veritas-chip {tone}">{escape(label)}: <strong>{escape(_format_value(value))}</strong></span>'


def _render_hero_summary(response, requested_top_k: int, effective_top_k: int, mode: str) -> str:
    verdict = escape(response.verdict)
    confidence = _format_value(response.confidence, 4)
    citation = "valid" if response.citation_valid else "needs review"
    backend = escape(response.backend_used)
    return f"""
    <div class="veritas-result-hero">
      <div class="veritas-chip-row">
        <span class="veritas-chip good">Verdict: <strong>{verdict}</strong></span>
        <span class="veritas-chip info">Confidence: <strong>{confidence}</strong></span>
        <span class="veritas-chip {'good' if response.citation_valid else 'warn'}">Citation: <strong>{citation}</strong></span>
        <span class="veritas-chip info">Mode: <strong>{escape(mode)}</strong></span>
      </div>
      <div class="veritas-result-grid">
        <div class="veritas-metric">
          <div class="metric-label">Requested top-k</div>
          <div class="metric-value">{requested_top_k}</div>
        </div>
        <div class="veritas-metric">
          <div class="metric-label">Effective top-k</div>
          <div class="metric-value">{effective_top_k}</div>
        </div>
        <div class="veritas-metric">
          <div class="metric-label">Latency</div>
          <div class="metric-value">{_format_value(response.latency_ms, 1)} ms</div>
        </div>
        <div class="veritas-metric">
          <div class="metric-label">Backend</div>
          <div class="metric-value">{backend}</div>
        </div>
      </div>
    </div>
    """


def _render_summary_strip(response, requested_top_k: int, effective_top_k: int) -> str:
    return f"""
    <div class="veritas-hero-side">
      <div class="veritas-signal">
        <h3>Live verifier status</h3>
        <p>The verifier is the source of truth. Retrieval, reranking, and explanation generation are surfaced separately so the gaps stay visible.</p>
      </div>
      <div class="veritas-rail">
        <div class="veritas-stat">
          <div class="label">Request ID</div>
          <div class="value">{escape(response.request_id)}</div>
        </div>
        <div class="veritas-stat">
          <div class="label">Citation validity</div>
          <div class="value">{'valid' if response.citation_valid else 'review'}</div>
        </div>
        <div class="veritas-stat">
          <div class="label">Fallback used</div>
          <div class="value">{'yes' if response.fallback_used else 'no'}</div>
        </div>
        <div class="veritas-stat">
          <div class="label">Retrieval window</div>
          <div class="value">{requested_top_k} → {effective_top_k}</div>
        </div>
      </div>
      <div class="veritas-note">
        Research mode widens the evidence window without hiding the measured retrieval-vs-oracle gap.
      </div>
    </div>
    """


def _render_backend_card(response) -> str:
    chips = " ".join(
        [
            _build_chip("Verifier", response.backend_used, "info"),
            _build_chip("Retriever", response.retrieval_backend, "info"),
            _build_chip("Reranker", response.reranker_backend, "info"),
            _build_chip("Explanation", response.explanation_mode, "info"),
        ]
    )
    return f"""
    <div class="veritas-card">
      <h4>Runtime metadata</h4>
      <div class="veritas-chip-row">{chips}</div>
      <div class="veritas-hr"></div>
      <p>Fallback flags: retrieval <strong>{'on' if response.retrieval_fallback_used else 'off'}</strong>, reranker <strong>{'on' if response.reranker_fallback_used else 'off'}</strong>, overall <strong>{'on' if response.fallback_used else 'off'}</strong>.</p>
      <p>Model: <strong>{escape(response.model_name)}</strong></p>
    </div>
    """


def _render_method_card() -> str:
    return """
    <div class="veritas-card">
      <h4>Method</h4>
      <p>Claim checks flow through deterministic preprocessing, BM25-first retrieval, dense or hybrid retrieval when available, cross-encoder reranking, transformer verification, and citation-grounded explanation generation.</p>
      <ul>
        <li>Verifier label is the final decision.</li>
        <li>Explanation quality is audited independently.</li>
        <li>Retrieval quality is measured separately from answer quality.</li>
      </ul>
    </div>
    """


def _render_limitations_card() -> str:
    return """
    <div class="veritas-card">
      <h4>Limitations</h4>
      <ul>
        <li>No SOTA claim.</li>
        <li>No production-scale benchmark claim.</li>
        <li>Oracle evidence still outperforms retrieved evidence materially.</li>
        <li>CUDA QLoRA and CUDA DPO are not part of the final stack.</li>
      </ul>
    </div>
    """


def _format_evidence_table(response) -> list[list[object]]:
    rows: list[list[object]] = []
    for item in response.evidence:
        rows.append(
            [
                item.citation_id,
                item.doc_id,
                item.title or "",
                item.score if item.score is not None else "",
                item.text,
            ]
        )
    return rows


def _render_evidence_summary(response) -> str:
    if not response.evidence:
        return """
        <div class="veritas-card">
          <h4>Evidence</h4>
          <p>No evidence returned for this claim.</p>
        </div>
        """

    top_titles = ", ".join(
        escape(item.title or item.doc_id) for item in response.evidence[:3]
    )
    return f"""
    <div class="veritas-card">
      <h4>Evidence snapshot</h4>
      <p>Top passages surfaced by the current retrieval stack: {top_titles}.</p>
      <p>The table below preserves the exact citations passed to the verifier.</p>
    </div>
    """


def _build_demo_note(mode: str, top_k: int, explanation_mode: str, selected_mode: str) -> str:
    return (
        f"Mode selected: {mode}. "
        f"Requested top-k: {top_k}. "
        f"Effective top-k: {selected_mode}. "
        f"Explanation lens: {explanation_mode}."
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Veritas") as demo:
        with gr.Column(elem_classes=["veritas-shell"]):
            gr.HTML(f"<style>{THEME_CSS}</style>")
            gr.HTML(
                """
                <div class="veritas-content">
                  <div class="veritas-hero">
                    <div class="veritas-hero-main">
                      <div>
                        <div class="veritas-kicker">Evidence-grounded fact verification</div>
                        <h1 class="veritas-title">Veritas</h1>
                        <div class="veritas-subtitle">
                          A local research and product-grade verification system for Mac workflows, with retrieval,
                          reranking, transformer verdicts, and citation-aware explanations exposed as a single audit trail.
                        </div>
                      </div>
                      <div class="veritas-hero-badges">
                        <span class="veritas-badge"><strong>Verifier first</strong> label selection is explicit</span>
                        <span class="veritas-badge"><strong>Hybrid retrieval</strong> quality is measured separately</span>
                        <span class="veritas-badge"><strong>Audit-ready</strong> final results are reproducible</span>
                      </div>
                    </div>
                    <div class="veritas-hero-side">
                      <div class="veritas-signal">
                        <h3>Research posture</h3>
                        <p>Honest measurement over hype. Oracle evidence, retrieved evidence, explanation faithfulness, and ranking quality all stay visible.</p>
                      </div>
                      <div class="veritas-rail">
                        <div class="veritas-stat">
                          <div class="label">Primary path</div>
                          <div class="value">BM25 + verifier</div>
                        </div>
                        <div class="veritas-stat">
                          <div class="label">Research path</div>
                          <div class="value">Hybrid + rerank</div>
                        </div>
                        <div class="veritas-stat">
                          <div class="label">Serving</div>
                          <div class="value">FastAPI + Gradio</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                """
            )

            with gr.Row(elem_classes=["veritas-grid"]):
                with gr.Column(elem_classes=["veritas-panel"]):
                    with gr.Row(elem_classes=["veritas-panel-title"]):
                        gr.Markdown("## Query")
                        gr.Markdown("Fast path defaults to the lowest-latency honest answer.")
                    claim = gr.Textbox(
                        label="Claim",
                        placeholder="Enter a factual claim to verify",
                        lines=4,
                    )
                    with gr.Row():
                        top_k = gr.Slider(1, 10, value=5, step=1, label="Top-k evidence")
                        mode = gr.Radio(
                            ["Fast Path", "Research Path"],
                            value="Fast Path",
                            label="Operating mode",
                        )
                    explanation_mode = gr.Dropdown(
                        choices=["Auto", "Template", "MLX LoRA", "Preference Reranked"],
                        value="Auto",
                        label="Explanation lens",
                        info="Display preference only; the backend selects the live explanation mode.",
                    )
                    verify_button = gr.Button("Verify claim", variant="primary")
                    gr.HTML(
                        """
                        <div class="veritas-card">
                          <h4>Examples</h4>
                          <p>Use these to sanity-check the pipeline and compare verdicts across a supported, refuted, and unknown claim.</p>
                        </div>
                        """
                    )
                    gr.Examples(
                        examples=_example_claims(),
                        inputs=[claim, top_k],
                        label=None,
                    )

                with gr.Column(elem_classes=["veritas-panel"]):
                    with gr.Row(elem_classes=["veritas-panel-title"]):
                        gr.Markdown("## Live output")
                        gr.Markdown("The verifier output stays visible with supporting metadata.")
                    verdict_panel = gr.HTML()
                    summary_strip = gr.HTML()

            with gr.Tabs():
                with gr.Tab("Results"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            explanation = gr.Textbox(label="Explanation", lines=8, interactive=False)
                            evidence_table = gr.Dataframe(
                                headers=["Citation", "Doc ID", "Title", "Score", "Text"],
                                datatype=["number", "str", "str", "number", "str"],
                                label="Evidence",
                                interactive=False,
                            )
                        with gr.Column(scale=1):
                            evidence_summary = gr.HTML()
                            backend_card = gr.HTML()
                            request_card = gr.HTML()

                with gr.Tab("Research"):
                    with gr.Row():
                        with gr.Column():
                            gr.HTML(_render_method_card())
                        with gr.Column():
                            gr.HTML(
                                """
                                <div class="veritas-card">
                                  <h4>Research focus</h4>
                                  <ul>
                                    <li>Retrieval ablation and reranking refinement.</li>
                                    <li>Oracle-vs-retrieved comparison.</li>
                                    <li>Top-k verifier evaluation to expose retrieval sensitivity.</li>
                                    <li>Final audit packaging for review and verification.</li>
                                  </ul>
                                </div>
                                """
                            )

                with gr.Tab("Limitations"):
                    with gr.Row():
                        with gr.Column():
                            gr.HTML(_render_limitations_card())
                        with gr.Column():
                            gr.HTML(
                                """
                                <div class="veritas-card">
                                  <h4>What this demo does not claim</h4>
                                  <ul>
                                    <li>SOTA benchmark performance.</li>
                                    <li>Perfect citation faithfulness.</li>
                                    <li>Full benchmark coverage beyond the measured sample-scale runs.</li>
                                    <li>CUDA-only training or adapter claims.</li>
                                  </ul>
                                </div>
                                """
                            )

            gr.Markdown(
                """
                <div class="veritas-footer">
                  Built for research review and product-grade demo use. The measured gap between oracle evidence and retrieved evidence is intentionally surfaced instead of hidden.
                </div>
                """
            )

        def submit(claim_text: str, top_k_value: int, selected_mode: str, selected_explanation_mode: str):
            requested_top_k = int(top_k_value)
            if selected_mode == "Research Path":
                effective_top_k = min(max(requested_top_k, 5), 10)
            else:
                effective_top_k = requested_top_k

            response = verify(VerifyRequest(claim=claim_text, top_k=effective_top_k))
            return (
                _render_hero_summary(response, requested_top_k, effective_top_k, selected_mode),
                _render_summary_strip(response, requested_top_k, effective_top_k),
                response.explanation,
                _format_evidence_table(response),
                _render_evidence_summary(response),
                _render_backend_card(response),
                f"""
                <div class="veritas-card">
                  <h4>Request details</h4>
                  <p>{escape(_build_demo_note(selected_mode, requested_top_k, selected_explanation_mode, effective_top_k))}</p>
                  <p>Request ID: <strong>{escape(response.request_id)}</strong></p>
                </div>
                """,
            )

        verify_button.click(
            submit,
            inputs=[claim, top_k, mode, explanation_mode],
            outputs=[
                verdict_panel,
                summary_strip,
                explanation,
                evidence_table,
                evidence_summary,
                backend_card,
                request_card,
            ],
        )

    return demo


if __name__ == "__main__":  # pragma: no cover - UI entrypoint
    build_demo().launch()
