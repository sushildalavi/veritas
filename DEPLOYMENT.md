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

## Lightweight vs. advanced serving modes

The backend (`serving/api.py`, `serving/model_loader.py`) reads its
configuration from `configs/serving.yaml` (or whatever `VERITAS_CONFIG`
points to), with every field overridable via environment variables. Two
modes are provided:

### Lightweight mode (default — `configs/serving.yaml`)

- Retrieval: `bm25_only` (no extra model downloads).
- Reranking: `none`.
- Verifier: auto-resolves to the clean class-weighted DistilRoBERTa
  checkpoint (`checkpoints/transformer_verifier_clean`, ~330MB,
  macro_f1≈0.711) if present, otherwise falls back to the sklearn
  TF-IDF/LogReg baseline (`checkpoints/verifier_clean`, <1MB,
  macro_f1≈0.484), otherwise the mock verifier.
- Suitable for free Hugging Face Spaces CPU tier: one transformer model in
  memory, no sentence-transformers/cross-encoder downloads.

### Advanced mode (`configs/serving_advanced.yaml`)

- Retrieval: `bm25_sentence_transformer_hybrid` — adds a dense retriever
  using `sentence-transformers/all-MiniLM-L6-v2` (~80MB) fused with BM25 via
  reciprocal rank fusion (see `reports/retrieval_eval_neural_large.md`:
  hybrid recall@10=0.535 vs. BM25-only 0.442).
- Reranking: `cross_encoder` — reranks retrieved candidates with
  `cross-encoder/ms-marco-MiniLM-L-6-v2` (~120MB) (see
  `reports/ranking_eval_cross_encoder_large.md`: cross-encoder ndcg@10=0.565
  vs. BM25 ndcg@10=0.389).
- Verifier: same auto-resolution as lightweight mode.
- Adds ~200MB of additional model downloads and extra per-request latency
  (dense embedding + cross-encoder scoring) versus lightweight mode. Use on
  a Space with more CPU/RAM, or when retrieval/ranking quality matters more
  than cold-start time and per-request latency.

To run advanced mode locally or in a Space:

```bash
VERITAS_CONFIG=configs/serving_advanced.yaml uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

or set the individual environment variables (`VERITAS_RETRIEVAL_BACKEND`,
`VERITAS_USE_NEURAL_RETRIEVAL`, `VERITAS_RERANKER_BACKEND`,
`VERITAS_USE_CROSS_ENCODER`) directly in the Space's settings.

`/health` and `/metrics` report the active `retrieval_backend`,
`reranker_backend`, `embedding_model`, `cross_encoder_model`, `model_name`,
and `verifier_macro_f1` so the running mode can be confirmed at runtime.
