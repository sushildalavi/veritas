"""Gradio UI stub for the Veritas demo."""

import gradio as gr


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Veritas") as demo:
        gr.Markdown("# Veritas\nClaim verification demo stub.")
    return demo


if __name__ == "__main__":
    build_demo().launch()
