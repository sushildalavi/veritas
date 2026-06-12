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
- MLX LoRA (Apple Silicon, `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, 100 iters, best adapter): verdict accuracy 0.695, macro F1 0.4632, citation valid rate 0.6 on 200 held-out examples (`reports/mlx_lora_eval_200.md`); a 300-iter adapter overfit and was rejected (`reports/mlx_lora_comparison.md`)
- DPO preference pairs: 1,382 `{"prompt", "chosen", "rejected"}` pairs built from `verifier_train.jsonl`, chosen-citation-valid rate 1.0 (`reports/preference_pair_stats.md`) — dataset construction only, no DPO training run
- CUDA QLoRA / DPO: **ready, not run** — no CUDA GPU / `bitsandbytes` on this machine; ready-to-run configs/scripts/notebooks checked in (`reports/cuda_qlora_READY.md`, `reports/cuda_dpo_READY.md`); MLX DPO also not available (`reports/mlx_dpo_READY.md`)

## Deployment Status

- `python3 app.py` launches the Gradio UI locally.
- `GET /health`, `POST /verify`, and `GET /metrics` are implemented and report the active verifier/retrieval/reranker backends.
- Public Hugging Face Spaces URL: https://sushildalavi-veritas.hf.space.
- Lightweight (default) and advanced serving configs documented in `DEPLOYMENT.md`.

## Completion Status

**PROJECT NOT COMPLETE: CUDA QLoRA and DPO have not been run.** All other phases (data, retrieval, ranking, verification, serving, faithfulness, Pareto analysis, docs, and Apple Silicon MLX LoRA alignment with 200-example evaluation) are complete with real, checked-in reports. A DPO preference-pair dataset (1,382 pairs) has been built. CUDA QLoRA and CUDA DPO Kaggle/Colab notebooks are ready but not run. See `reports/final_completion_gate.md` for the full checklist.

## Safe Resume Line

Veritas is a production-oriented fact verification portfolio project with reproducible sample data, real large-scale evaluation reports, a class-weighted transformer verifier as the default serving backend, a trained Apple Silicon MLX LoRA adapter (best of 100/300-iter runs, evaluated on 200 examples) for citation-grounded verification, a 1,382-pair DPO preference dataset, and ready-to-run (but not yet executed) CUDA QLoRA and CUDA DPO Kaggle/Colab notebooks.
