# Final Project Audit

## Scope Completed

- Phase 1: production config, artifact manifest, and environment discipline
- Phase 2: real dense retrieval backend and retrieval evaluation
- Phase 3: optional cross-encoder reranking and ranking evaluation
- Phase 4: transformer verifier fine-tuning path and checkpoint
- Phase 5: serving hardening, validation, monitoring, and backend metadata
- Phase 6: Hugging Face Spaces deployment readiness and Gradio launch path
- Phase 7: QLoRA and DPO left as offline optional extensions
- Phase 8: README and project summary rewritten with real metrics only
- Phase 9: final completion summary prepared for review

## Generated Artifacts

- `data/processed/evidence_corpus.jsonl`
- `data/processed/fever_train.jsonl`
- `data/processed/fever_val.jsonl`
- `data/processed/fever_test.jsonl`
- `data/processed/scifact_train.jsonl`
- `data/processed/scifact_val.jsonl`
- `data/processed/scifact_test.jsonl`
- `reports/data_quality.json`
- `reports/retrieval_eval.json`
- `reports/retrieval_eval_neural.json`
- `reports/ranking_eval.json`
- `reports/ranking_eval_cross_encoder.json`
- `reports/verifier_training.json`
- `reports/transformer_verifier_eval.json`
- `reports/faithfulness_eval.json`
- `reports/error_analysis_summary.json`
- `reports/pareto_analysis.json`
- `reports/artifact_manifest.json`
- `reports/final_project_audit.md`
- `reports/final_completion_summary.md`
- `checkpoints/verifier/model.joblib`
- `checkpoints/transformer_verifier/config.json`
- `checkpoints/transformer_verifier/model.safetensors`
- `checkpoints/transformer_verifier/tokenizer.json`
- `checkpoints/transformer_verifier/tokenizer_config.json`
- `checkpoints/transformer_verifier/training_args.bin`

## Validation

- `python3 -m pytest` passes with 58 tests.
- `python3 -m compileall` passes for the codebase.
- `make test`, `make lint`, and `make all-evals` pass in the current environment.
- `python3 app.py` starts the Gradio demo entrypoint locally.

## Key Measured Results

- Data quality: 8 sampled records, 1 missing evidence span, 0 duplicate claims.
- Retrieval baseline: 12-passage evidence corpus, FEVER validation BM25 MRR of 1.000.
- Neural retrieval: sentence-transformers backend, dense Recall@1 of 0.750, dense MRR of 1.000.
- Ranking baseline: learned ranker backend `sklearn-logistic`, learned MAP of 0.271.
- Cross-encoder ranking: cross-encoder MAP of 0.667, MRR of 0.667.
- Sklearn verifier: train accuracy 0.750, validation accuracy 0.400, test accuracy 0.333.
- Transformer verifier smoke run: train accuracy 0.500, validation accuracy 0.000, test accuracy 0.000.
- Faithfulness: citation validity rate 0.875, verdict consistency rate 0.875.
- Error analysis: 6 mismatches out of 8 records, error rate 0.750.
- Pareto frontier: `mock-top5` with macro-F1 0.302 and latency 0.0676 ms.

## Notes

- All metrics are sample-scale and are useful for tracking regressions, not for broad benchmark claims.
- No fabricated metrics are introduced in the README or this audit.
- The public Hugging Face Spaces URL is live at `https://sushildalavi-veritas.hf.space`.
- QLoRA and DPO are documented as offline optional extensions only.
