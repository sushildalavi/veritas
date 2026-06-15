# Transformer Verifier (Clean Dataset) Evaluation

- Model: `distilroberta-base`
- Checkpoint: `checkpoints/transformer_verifier_robust`
- Epochs: 2.0
- Batch size: 32
- Learning rate: 2e-05
- Training runtime (s): 1008.563
- Git commit: a9a721d2faa0990707a1113de0cdbf7d178c5e86
- Training command: `python3 scripts/train_transformer_verifier_clean.py --train-file data/processed/verifier_train_robust.jsonl --output-dir checkpoints/transformer_verifier_robust --report-json reports/transformer_verifier_robust_eval.json --report-md reports/transformer_verifier_robust_eval.md --batch-size 32 --epochs 2`

| split | examples | accuracy | macro_f1 |
| --- | --- | --- | --- |
| train | 6471 | 0.728 | 0.510 |
| validation | 649 | 0.641 | 0.498 |
| test | 642 | 0.620 | 0.493 |

## Test set per-class metrics

| label | precision | recall | f1 |
| --- | --- | --- | --- |
| SUPPORTED | 0.511 | 0.839 | 0.635 |
| REFUTED | 0.000 | 0.000 | 0.000 |
| NOT_ENOUGH_INFO | 0.766 | 0.942 | 0.845 |

## Test set confusion matrix

| true \ pred | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
| --- | --- | --- | --- |
| SUPPORTED | 188 | 0 | 36 |
| REFUTED | 167 | 0 | 28 |
| NOT_ENOUGH_INFO | 13 | 0 | 210 |

## Threshold check

- macro_f1 >= 0.55: FAIL (0.493)
- accuracy >= 0.60: PASS (0.620)

