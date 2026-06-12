# Veritas Project Summary

Veritas is a sample-scale but end-to-end factual claim verification system built around FEVER and SciFact artifacts.
It includes retrieval, ranking, verification, citation faithfulness evaluation, API serving, and a Gradio demo.

## What Exists

- Sampled processed datasets in `data/processed/` (2809/650/650 train/val/test verifier examples; 9,804-passage evidence corpus)
- Checked-in benchmark reports in `reports/`
- Sklearn TF-IDF/LogReg verifier checkpoint in `checkpoints/verifier_clean/` (macro F1 0.484)
- Class-weighted DistilRoBERTa verifier checkpoint in `checkpoints/transformer_verifier_clean/` (macro F1 0.711, REFUTED recall 0.745) — the default serving backend
- Large-scale (200-query) neural retrieval and cross-encoder ranking reports
- FastAPI service and Gradio demo with config-driven backend routing

## Real Metrics

- Data quality (large pipeline): 4109 sampled records, see `reports/data_quality_large.md`
- Sklearn verifier (clean): test accuracy 0.486, macro F1 0.484 (`reports/verifier_clean_baseline.md`)
- DistilRoBERTa verifier (clean, class-weighted): test accuracy 0.718, macro F1 0.711, REFUTED recall 0.745 (`reports/transformer_verifier_clean_eval.md`)
- Oracle vs. retrieved: oracle macro F1 0.710 vs. end-to-end (BM25 top-1) macro F1 0.414 (`reports/oracle_verifier_eval.md`, `reports/end_to_end_verifier_eval.md`)
- Neural retrieval (200 queries): dense recall@1 0.357, hybrid recall@10 0.535 (`reports/retrieval_eval_neural_large.md`)
- Cross-encoder ranking (200 queries): MAP 0.540, MRR 0.562, nDCG@10 0.565 (`reports/ranking_eval_cross_encoder_large.md`)
- Faithfulness (template explanations, 200 val examples): citation validity 0.560, verdict consistency 0.755 (`reports/faithfulness_comparison.md`)
- Pareto (measured): `distilroberta-clean` macro F1 0.711 / 316.7MB vs. `sklearn-tfidf-logreg` macro F1 0.484 / 0.6MB (`reports/final_pareto_analysis.md`)
- QLoRA / DPO: **not trained** — no CUDA GPU / `bitsandbytes` available (`reports/qlora_BLOCKED_GPU_REQUIRED.md`, `reports/dpo_BLOCKED_QLORA_REQUIRED.md`)

## Deployment Status

- `python3 app.py` launches the Gradio UI locally.
- `GET /health`, `POST /verify`, and `GET /metrics` are implemented and report the active verifier/retrieval/reranker backends.
- Public Hugging Face Spaces URL: https://sushildalavi-veritas.hf.space.
- Lightweight (default) and advanced serving configs documented in `DEPLOYMENT.md`.

## Completion Status

**PROJECT NOT COMPLETE: QLoRA and DPO require GPU execution.** All other phases (data, retrieval, ranking, verification, serving, faithfulness, Pareto analysis, docs) are complete with real, checked-in reports. See `reports/final_completion_gate.md` for the full checklist.

## Safe Resume Line

Veritas is a production-oriented fact verification portfolio project with reproducible sample data, real large-scale evaluation reports, a class-weighted transformer verifier as the default serving backend, and ready-to-run (but not yet executed) QLoRA/DPO training scripts pending GPU access.
