# Verifier Threshold Calibration

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Retrieval backend: `bm25_only`
- Reranker backend: `none`
- Split prefixes: `fever_val, scifact_val`
- Sample size: 20
- Top-k retrieved evidence: 5

## Summary

- Baseline thresholds: support=0.5, refute=0.5
- Baseline macro_f1: 0.2692
- Best thresholds: support=0.55, refute=0.3
- Best macro_f1: 0.3974

## Top candidates

| support | refute | accuracy | macro_f1 | nei_fp_rate |
| --- | --- | --- | --- | --- |
| 0.55 | 0.3 | 0.4 | 0.3974 | 0.5 |
| 0.55 | 0.35 | 0.4 | 0.3974 | 0.5 |
| 0.55 | 0.4 | 0.4 | 0.3974 | 0.5 |
| 0.55 | 0.45 | 0.4 | 0.3974 | 0.5 |
| 0.55 | 0.5 | 0.35 | 0.3488 | 0.5 |
| 0.6 | 0.3 | 0.35 | 0.3373 | 0.5 |
| 0.6 | 0.35 | 0.35 | 0.3373 | 0.5 |
| 0.6 | 0.4 | 0.35 | 0.3373 | 0.5 |
| 0.6 | 0.45 | 0.35 | 0.3373 | 0.5 |
| 0.3 | 0.3 | 0.4 | 0.3095 | 1.0 |
