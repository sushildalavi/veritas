# Transformer Verifier (Clean Dataset) Evaluation

- Model: `distilroberta-base`
- Checkpoint: `checkpoints/transformer_verifier_clean`
- Epochs: 2.0
- Batch size: 32
- Learning rate: 2e-05
- Training runtime (s): 538.888
- Git commit: cb758630f1a1251291acf0f391784eb6f368dab9
- Training command: `python3 scripts/train_transformer_verifier_clean.py --batch-size 32 --epochs 2`

| split | examples | accuracy | macro_f1 |
| --- | --- | --- | --- |
| train | 2809 | 0.713 | 0.722 |
| validation | 650 | 0.715 | 0.701 |
| test | 650 | 0.718 | 0.711 |

## Test set per-class metrics

| label | precision | recall | f1 |
| --- | --- | --- | --- |
| SUPPORTED | 0.667 | 0.469 | 0.551 |
| REFUTED | 0.562 | 0.745 | 0.641 |
| NOT_ENOUGH_INFO | 0.938 | 0.946 | 0.942 |

## Test set confusion matrix

| true \ pred | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
| --- | --- | --- | --- |
| SUPPORTED | 106 | 115 | 5 |
| REFUTED | 42 | 149 | 9 |
| NOT_ENOUGH_INFO | 11 | 1 | 212 |

## Threshold check

- macro_f1 >= 0.55: PASS (0.711)
- accuracy >= 0.60: PASS (0.718)

