# Transformer Verifier Evaluation

- Model: `distilroberta-base`
- Checkpoint: `checkpoints/transformer_verifier`
- Train examples: 2
- Validation examples: 1
- Test examples: 1
- Test latency ms per example: 18.16

| split | examples | accuracy | macro_f1 |
| --- | --- | --- | --- |
| train | 2 | 0.500 | 0.333 |
| validation | 1 | 0.000 | 0.000 |
| test | 1 | 0.000 | 0.000 |

## Per-class test metrics

| label | precision | recall | f1 |
| --- | --- | --- | --- |
| SUPPORTED | 0.000 | 0.000 | 0.000 |
| REFUTED | 0.000 | 0.000 | 0.000 |
| NOT_ENOUGH_INFO | 0.000 | 0.000 | 0.000 |

## Confusion matrix

```text
0 0 1
0 0 0
0 0 0
```
