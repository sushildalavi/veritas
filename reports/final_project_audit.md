# Final Project Audit

## Scope Completed

- Phase A: repository cleanup and artifact directories
- Phase B: sampled FEVER and SciFact data pipeline
- Phase C: retrieval evaluation
- Phase D: ranking evaluation
- Phase E: lightweight verifier training
- Phase F: checkpoint-backed serving defaults
- Phase G: faithfulness and citation evaluation
- Phase H: error analysis and Pareto analysis
- Phase I: Spaces entrypoint and demo launch files
- Phase J: README refresh with real metrics

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
- `reports/ranking_eval.json`
- `reports/verifier_training.json`
- `reports/faithfulness_eval.json`
- `reports/error_analysis_summary.json`
- `reports/error_analysis.json`
- `reports/pareto_analysis.json`
- `reports/final_project_audit.md`
- `checkpoints/verifier/model.joblib`

## Validation

- `python3 -m pytest` passed for the sampled pipeline, benchmark loaders, verifier checkpoint loader, API health contract, Pareto helper, and app entrypoint tests.
- `python3 -m compileall models scripts evaluation retrieval ranking data training rag agent serving ui tests` completed successfully.
- `make eval-retrieval`, `make eval-ranking`, `make eval-faithfulness`, `make error-analysis`, and `make pareto-analysis` all produced reports from real local runs.

## Key Measured Results

- Data quality: 8 sampled records, 1 missing evidence span, 0 duplicate claims.
- Retrieval: 12-passage evidence corpus, FEVER validation BM25 MRR of 1.000.
- Ranking: learned ranker backend `sklearn-logistic`, FEVER train learned MAP of 0.750.
- Verification: train accuracy 0.750, validation accuracy 0.400, test accuracy 0.333.
- Faithfulness: citation validity rate 0.875, verdict consistency rate 0.875.
- Error analysis: 6 mismatches out of 8 records, error rate 0.750.
- Pareto frontier: `mock-top5` with macro-F1 0.302 and latency 0.20 ms.

## Notes

- These values are from a small sampled run and are suitable for regression tracking, not broad model claims.
- No fabricated metrics are introduced in the README or this audit.
- The trained checkpoint is stored locally under `checkpoints/verifier/` and is preferred automatically when present.
