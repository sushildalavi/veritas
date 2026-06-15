# Verifier Threshold Calibration

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Git commit: `a21ede24d7b0f051091cb5405c50080ad46e0838`
- Config path: `configs/serving.yaml`
- Dataset source: `structured records from fever_val_large, scifact_val_large`
- Retrieval backend: `bm25_only`
- Reranker backend: `none`
- Split prefixes: `fever_val, scifact_val`
- Sample size: 100
- Top-k retrieved evidence: 5

## Summary

- Baseline thresholds: support=0.55, refute=0.3
- Baseline macro_f1: 0.4245
- Best thresholds: support=0.55, refute=0.5
- Best macro_f1: 0.4388

## Top candidates

| support | refute | accuracy | macro_f1 | nei_fp_rate |
| --- | --- | --- | --- | --- |
| 0.55 | 0.5 | 0.45 | 0.4388 | 0.6061 |
| 0.55 | 0.3 | 0.44 | 0.4245 | 0.6667 |
| 0.55 | 0.35 | 0.44 | 0.4245 | 0.6667 |
| 0.55 | 0.4 | 0.44 | 0.4245 | 0.6667 |
| 0.55 | 0.45 | 0.44 | 0.4245 | 0.6667 |
| 0.4 | 0.5 | 0.43 | 0.3826 | 0.9091 |
| 0.45 | 0.5 | 0.42 | 0.3748 | 0.9091 |
| 0.5 | 0.5 | 0.42 | 0.3748 | 0.9091 |
| 0.7 | 0.5 | 0.47 | 0.3649 | 0.2424 |
| 0.75 | 0.5 | 0.47 | 0.3649 | 0.2424 |
