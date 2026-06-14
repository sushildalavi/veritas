# Transformer Verifier (Clean Dataset) Evaluation

- Model: `microsoft/deberta-v3-xsmall`
- Checkpoint: `checkpoints/deberta_verifier_clean`
- Epochs: 1.0
- Batch size: 8
- Learning rate: 2e-05
- Training runtime (s): 297.472
- Git commit: 96980b27b3182adefad622764942da4b26e564ce
- Training command: `python scripts/train_transformer_verifier_clean.py --model-name microsoft/deberta-v3-xsmall --output-dir checkpoints/deberta_verifier_clean --report-json reports/deberta_verifier_clean_eval.json --report-md reports/deberta_verifier_clean_eval.md --epochs 1 --batch-size 8 --learning-rate 2e-5 --max-length 192 --gradient-accumulation-steps 1 --warmup-ratio 0.06 --weight-decay 0.01 --use-class-weights --use-cpu --seed 42`

| split | examples | accuracy | macro_f1 |
| --- | --- | --- | --- |
| train | 2808 | 0.725 | 0.574 |
| validation | 649 | 0.653 | 0.536 |
| test | 642 | 0.636 | 0.537 |

## Test set per-class metrics

| label | precision | recall | f1 |
| --- | --- | --- | --- |
| SUPPORTED | 0.500 | 0.960 | 0.657 |
| REFUTED | 0.438 | 0.036 | 0.066 |
| NOT_ENOUGH_INFO | 0.949 | 0.834 | 0.888 |

## Test set confusion matrix

| true \ pred | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
| --- | --- | --- | --- |
| SUPPORTED | 215 | 3 | 6 |
| REFUTED | 184 | 7 | 4 |
| NOT_ENOUGH_INFO | 31 | 6 | 186 |

## Threshold check

- macro_f1 >= 0.55: FAIL (0.537)
- accuracy >= 0.60: PASS (0.636)

