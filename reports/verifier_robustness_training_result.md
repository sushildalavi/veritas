# Verifier Robustness Training — Measured Results

## Objective

Retrain the DistilRoBERTa verifier on a combined oracle + retrieved-evidence
training set to improve robustness to noisy retrieved passages. The baseline
verifier (`checkpoints/transformer_verifier_clean`) was trained only on oracle
(claim, gold-evidence) pairs; it sees a large performance drop on retrieved
evidence (oracle per_passage_max macro-F1 0.6728 → retrieved 0.3887 on the
full 650-example v2 test set).

## Method

- Combined `data/processed/verifier_train.jsonl` (2808 oracle examples) with
  `data/processed/retrieved_evidence_verifier_calibration.jsonl` (3663
  retrieved-evidence pairs, mapped SUPPORTS/REFUTES/NEI → SUPPORTED/REFUTED/
  NOT_ENOUGH_INFO) into `data/processed/verifier_train_robust.jsonl` (6471
  total examples).
- Same model, optimizer, and hyperparameters as the baseline
  (`distilroberta-base`, batch_size=32, epochs=2, lr=2e-5), to isolate the
  effect of the training data alone.
- New checkpoint: `checkpoints/transformer_verifier_robust`.
- Evaluated with `scripts/eval_oracle_vs_retrieved_v2.py --max-examples 650`.

## Dataset composition

| source | SUPPORTED | REFUTED | NOT_ENOUGH_INFO | total |
| --- | --- | --- | --- | --- |
| oracle (verifier_train.jsonl) | 1416 | 604 | 788 | 2808 |
| retrieved-evidence (calibration) | 499 | 395 | 2769 | 3663 |
| combined | 1915 | 999 | 3557 | 6471 |

Note: the retrieved-evidence calibration set is 76% NEI, because most
retrieved passages (irrelevant, near-miss, same_topic_missing_fact) are
correctly labeled NOT_ENOUGH_INFO. Combined, NEI is 55% of the training set
vs 28% for oracle alone.

## Results (full 650-example v2 test set)

| checkpoint | evidence | mode | accuracy | macro_f1 |
| --- | --- | --- | --- | --- |
| transformer_verifier_clean (baseline) | oracle | per_passage_max | 0.6154 | 0.6728 |
| transformer_verifier_clean (baseline) | retrieved | per_passage_max | 0.4246 | 0.3887 |
| transformer_verifier_robust (this run) | oracle | per_passage_max | 0.5615 | 0.4376 |
| transformer_verifier_robust (this run) | retrieved | per_passage_max | 0.3862 | 0.2829 |

## Negative result

The robust checkpoint **regressed** on both oracle and retrieved evidence.

- Retrieved per_passage_max macro_F1: 0.3887 → 0.2829 (−0.1058)
- Oracle per_passage_max macro_F1: 0.6728 → 0.4376 (−0.2352)

The model learned to be too conservative — the heavy NEI skew in the
retrieved-evidence augmentation (76% NEI) trained the model to predict NEI
excessively, hurting accuracy on SUPPORTED and REFUTED examples even with
oracle evidence.

## Why it failed

1. **Class imbalance**: adding 2769 NEI pairs to 788 existing NEI examples
   shifted the combined set to 55% NEI. Without class weights or balanced
   sampling, the model collapses toward NEI.
2. **Domain mismatch in negatives**: the retrieved-evidence negatives are
   labeled per-passage. The val/test sets remain oracle-only, so the model is
   evaluated in a regime it has been trained to avoid (predicting
   SUPPORTED/REFUTED for gold evidence).
3. **No class reweighting**: the baseline run used no class weights and no
   weighted sampler; neither was applied here for a clean comparison, but
   class reweighting would likely be required to use this augmentation
   productively.

## What was not tried

- `--use-class-weights` or `--use-weighted-sampler` (would counteract the NEI
  skew; likely required for a productive robustness retrain).
- Capping the number of NEI augmentation pairs (e.g. match the SUPPORTED
  count in the retrieved-evidence set).
- Mixing retrieved-evidence positives only (positive_oracle + positive_retrieved
  pair types) with oracle examples, without the NEI negatives.

## Conclusion

Naive augmentation with the full retrieved-evidence calibration set (76% NEI)
regressed full-set macro-F1. The robustness training direction is sound, but
it requires class reweighting or NEI-capping to produce a net improvement.

`checkpoints/transformer_verifier_clean` remains the production checkpoint.
No configs were changed to point to the regressed checkpoint.
