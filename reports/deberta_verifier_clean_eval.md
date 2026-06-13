# DeBERTa Challenger Evaluation

- checkpoint: checkpoints/deberta_verifier_clean
- baseline_checkpoint: checkpoints/transformer_verifier_clean
- model_size_mb: 143.09

| split | model | accuracy | macro_f1 | refuted_recall | mean_latency_seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| eval | challenger | 0.51 | 0.2252 | 0.0 | 0.0347 |
| eval | baseline | 0.545 | 0.368 | 0.4592 | 0.0307 |
| test | challenger | 0.52 | 0.2281 | 0.0 | 0.041 |
| test | baseline | 0.645 | 0.4383 | 0.625 | 0.0282 |
