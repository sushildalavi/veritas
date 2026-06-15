# Retrieved-Evidence Per-Passage Verifier Eval (current vs relevance-gated)

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Git commit: `0e94481e6a1977641d9fc3365fd07682dc00d823`
- Calibration pairs: 3663, holdout pairs: 3676
- Gate grid: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
- Best gate threshold (tuned on calibration): 0.5 (calibration macro_f1: current=0.3118, gated=0.4221)

## Holdout (650-claim test set, per-passage pairs)

| predictor | split | accuracy | macro_f1 | SUPPORTS F1 | REFUTES F1 | NEI F1 | support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current_on_holdout | overall | 0.3368 | 0.3192 | 0.2451 | 0.3175 | 0.395 | 3676 |
| current_on_holdout | fever | 0.3681 | 0.3541 | 0.2515 | 0.3849 | 0.4259 | 2825 |
| current_on_holdout | scifact | 0.2327 | 0.2234 | 0.1991 | 0.1836 | 0.2876 | 851 |
| gated_on_holdout | overall | 0.6785 | 0.4511 | 0.231 | 0.3026 | 0.8196 | 3676 |
| gated_on_holdout | fever | 0.6963 | 0.4752 | 0.2561 | 0.3413 | 0.8283 | 2825 |
| gated_on_holdout | scifact | 0.6193 | 0.3734 | 0.1103 | 0.2202 | 0.7895 | 851 |

## Delta (gated vs current, holdout overall)

- macro_f1 delta: 0.1319
- accuracy delta: 0.3417

**Verdict: per-passage holdout macro_f1 improved by 0.1319 (0.3192 -> 0.4511)**

_Note: this is a per-passage diagnostic benchmark, separate from the official 650-claim oracle-vs-retrieved evaluation. A gain here does not by itself constitute a full-650-set improvement._
