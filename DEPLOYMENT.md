# Deployment

Veritas is set up for a free Hugging Face Spaces deployment using the Gradio app in `app.py`.

## Runtime

- Space SDK: `Gradio`
- Python version: `3.11` from [`runtime.txt`](runtime.txt)
- Entry point: [`app.py`](app.py)
- Dependencies: [`requirements.txt`](requirements.txt)

## Steps

1. Create a new Hugging Face Space.
1. Choose `Gradio` as the SDK.
1. Set Python to `3.11`.
1. Connect the Space to this GitHub repository or upload the repo contents.
1. Confirm the Space starts by running `app.py` directly.
1. Verify the app shows a claim form, verdict, confidence, evidence, citation status, backend, and fallback status.
1. If the Space gets a public URL, add that URL to [`README.md`](README.md).

## Notes

- The demo uses the local fallback verifier when a transformer checkpoint is not available.
- No API server needs to be started separately for the Gradio demo.
- The app is designed to run on CPU-only free infrastructure.
- Keep secrets out of the Space; the demo does not require API tokens.
