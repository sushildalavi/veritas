# Veritas Project Summary

Veritas is a sample-scale but end-to-end factual claim verification system built around FEVER and SciFact artifacts.
It includes retrieval, ranking, verification, citation faithfulness evaluation, API serving, and a Gradio demo.

## What Exists

- Sampled processed datasets in `data/processed/`
- Checked-in benchmark reports in `reports/`
- Lightweight sklearn verifier checkpoint in `checkpoints/verifier/`
- Transformer verifier smoke checkpoint in `checkpoints/transformer_verifier/`
- Optional neural retrieval and cross-encoder ranking reports
- FastAPI service and Gradio demo

## Real Metrics

- Data quality: 8 sampled records, 1 missing evidence span, 0 duplicates
- Retrieval baseline: 12-passage corpus, BM25 validation MRR 1.000
- Neural retrieval: sentence-transformers dense Recall@1 0.750, MRR 1.000
- Ranking baseline: learned MAP 0.271, RRF MAP 0.750
- Cross-encoder ranking: MAP 0.667, MRR 0.667
- Sklearn verifier: train/val/test accuracy 0.750 / 0.400 / 0.333
- Transformer verifier smoke run: train/val/test accuracy 0.500 / 0.000 / 0.000
- Faithfulness: citation validity 0.875, verdict consistency 0.875
- Pareto frontier: `mock-top5`, macro-F1 0.302, latency 0.068 ms

## Deployment Status

- `python3 app.py` launches the Gradio UI locally.
- `GET /health`, `POST /verify`, and `GET /metrics` are implemented.
- Public Hugging Face Spaces URL: https://sushildalavi-veritas.hf.space.

## Safe Resume Line

Veritas is a production-oriented fact verification portfolio project with reproducible sample data, real evaluation reports, and both lightweight and transformer-based model paths.
