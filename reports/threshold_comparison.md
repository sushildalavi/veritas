# Threshold Comparison

- Git commit: `a21ede24d7b0f051091cb5405c50080ad46e0838`
- Config path: `configs/serving.yaml`
- Dataset source: `structured records from fever_test_large, scifact_test_large`
- Sample size: 100

| setting | support | refute | bundle_macro_f1 | per_passage_macro_f1 | per_passage_nei_fp_rate |
| --- | --- | --- | --- | --- | --- |
| default | 0.5 | 0.5 | 0.3901 | 0.4409 | 0.8065 |
| calibrated | 0.55 | 0.5 | 0.3901 | 0.4615 | 0.6129 |

## Delta

- bundle_macro_f1: 0.0
- per_passage_macro_f1: 0.0206
- per_passage_nei_false_positive_rate: -0.1936
