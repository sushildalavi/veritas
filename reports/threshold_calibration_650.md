# Retrieved-Evidence Threshold Calibration

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Git commit: `c7757641e9da0b5c1c2f6501541b738c9144a57c`
- Config path: `configs/serving.yaml`
- Calibration split: `structured records from fever_val_large, scifact_val_large` (650 examples)
- Held-out split: `structured records from fever_test_large, scifact_test_large` (650 examples)
- Retrieval backend: `bm25_only`

## Calibration grid search

- Grid: support=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], refute=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], margin=[0.0, 0.02, 0.05, 0.1]
- Default thresholds (support=0.55, refute=0.5, margin=0.0): calibration macro_f1=0.384
- Best on calibration: support=0.55, refute=0.5, margin=0.0, macro_f1=0.384

## Held-out (650-example test set) comparison

| setting | support | refute | margin | accuracy | macro_f1 | SUPPORTED F1 | REFUTED F1 | NEI F1 | refuted_pred_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_on_holdout | 0.55 | 0.5 | 0.0 | 0.4062 | 0.3887 | 0.3211 | 0.4974 | 0.3476 | 0.5708 |
| best_on_holdout | 0.55 | 0.5 | 0.0 | 0.4062 | 0.3887 | 0.3211 | 0.4974 | 0.3476 | 0.5708 |

## Delta (best vs. default, held-out)

- macro_f1 delta: 0.0
- accuracy delta: 0.0
- refuted_prediction_rate delta: 0.0

**Verdict: no change in held-out macro_f1; current thresholds remain best**
