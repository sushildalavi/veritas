# Veritas Training Artifacts — Summary

This report consolidates all training and evaluation work from the
`veritas-training-artifacts` branch. Results are measured on the full
650-example v2 test set unless noted.

---

## Baseline (transformer_verifier_clean)

The production verifier is `checkpoints/transformer_verifier_clean`
(DistilRoBERTa, trained on oracle evidence pairs).

| mode | evidence | accuracy | macro_f1 | nei_fpr |
| --- | --- | --- | --- | --- |
| per_passage_max | oracle | 0.6985 | 0.6728 | 0.0 |
| per_passage_max | retrieved | 0.4062 | 0.3887 | 0.7098 |

The oracle→retrieved macro_F1 gap is **0.2841**. NEI false-positive rate on
retrieved evidence is 0.7098 — the verifier assigns NOT_ENOUGH_INFO to most
retrieved passages because they are genuinely non-gold evidence.

---

## Phase 3 — Verifier Robustness Retrain

**Goal:** reduce the oracle→retrieved macro_F1 gap by augmenting training with
retrieved-evidence pairs.

**Method:** combined 2808 oracle examples + 3663 retrieved-evidence pairs
(from calibration set) into 6471 examples. Mapped SUPPORTS/REFUTES/NEI →
SUPPORTED/REFUTED/NOT_ENOUGH_INFO. Same distilroberta-base hyperparameters.

**Result: regression — production checkpoint unchanged.**

| checkpoint | evidence | accuracy | macro_f1 |
| --- | --- | --- | --- |
| transformer_verifier_clean (baseline) | oracle | 0.6154 | 0.6728 |
| transformer_verifier_clean (baseline) | retrieved | 0.4246 | 0.3887 |
| transformer_verifier_robust (this run) | oracle | 0.5615 | 0.4376 |
| transformer_verifier_robust (this run) | retrieved | 0.3862 | 0.2829 |

The robust checkpoint regressed on both oracle (−0.2352 macro_F1) and
retrieved evidence (−0.1058 macro_F1).

**Root cause:** the retrieved-evidence calibration set is 76% NEI, making
the combined set 55% NEI without class reweighting. The model collapsed toward
predicting NEI on everything.

**What was not tried:** class weights / weighted sampler; capping NEI pairs;
using only positive retrieved pairs. These are the obvious next-attempt levers.

Details: `reports/verifier_robustness_training_result.md`

---

## Phase 4 — Relevance Gate

**Goal:** filter low-overlap passages before verifier scoring to reduce the
NEI false-positive rate on retrieved evidence.

**Method:** lexical token overlap gate (fraction of non-trivial claim tokens
present in passage). Gate threshold calibrated on per-passage diagnostic data,
applied at prediction time. Gate is config-driven (`relevance_gate_threshold`
in `configs/serving.yaml`), disabled by default (`null`).

**Result: NEI FPR improved but macro_F1 regressed — gate stays disabled.**

| mode | evidence | accuracy | macro_f1 | nei_fpr |
| --- | --- | --- | --- | --- |
| per_passage_max (baseline, no gate) | retrieved | 0.4062 | 0.3887 | 0.7098 |
| per_passage_max (gate=0.5) | retrieved | 0.3557 | 0.3557 | 0.2857 |

NEI false-positive rate: 0.7098 → 0.2857 (−0.4241) — gate works as intended
for filtering irrelevant passages.
Retrieved macro_F1: 0.3887 → 0.3557 (−0.0330) — but gating also removes
some genuine evidence, causing a net regression on the primary metric.

The threshold was calibrated on per-passage pair data, not claim-level
aggregation; the transfer does not hold without separate calibration.

**Gate remains available:** set `relevance_gate_threshold: 0.5` in
`configs/serving.yaml` or `VERITAS_RELEVANCE_GATE_THRESHOLD=0.5` to enable.
Do not enable by default until a threshold is calibrated on claim-level data
and shows a macro_F1 improvement on the full test set.

Details: `reports/oracle_vs_retrieved_v2_full_gated.md`

---

## Phases 5–8 — Phi-3 QLoRA and DPO (Explanation Model)

These phases target the explanation generation model, not the verdict
classifier. The verifier still decides the label; QLoRA/DPO only affect the
quality of generated explanations.

**Status: blocked — CUDA unavailable on this Mac.**

No checkpoint or metric was fabricated. Skipped reports are at:
- `reports/phi3_qlora_skipped_or_training_metrics.md`
- `reports/phi3_dpo_skipped_or_training_metrics.md`

### What exists

| artifact | path | status |
| --- | --- | --- |
| SFT training set | `data/explanations/sft_train.jsonl` | exists |
| SFT val set | `data/explanations/sft_val.jsonl` | exists |
| DPO preference train | `data/explanations/dpo_train.jsonl` | exists |
| DPO preference val | `data/explanations/dpo_val.jsonl` | exists |
| QLoRA config | `configs/phi3_qlora.yaml` | exists |
| DPO config | `configs/phi3_dpo.yaml` | exists |
| QLoRA training script | `scripts/train_phi3_qlora.py` | exists |
| DPO training script | `scripts/train_phi3_dpo.py` | exists |
| Colab notebook | `notebooks/phi3_colab_training.ipynb` | exists |

### Dry-run verification (run locally)

QLoRA preflight (`python3 scripts/train_phi3_qlora.py --dry-run`):
all 4 checks pass (train file, eval file, output parent, base model).

DPO preflight (`python3 scripts/train_phi3_dpo.py --dry-run`):
4 of 5 checks pass; `adapter_exists: false` for `adapters/phi3_veritas_qlora`
is expected — the adapter doesn't exist until QLoRA training completes on GPU.

### To train on Colab T4

Open `notebooks/phi3_colab_training.ipynb`. The notebook:
- Runs QLoRA cells first (Q1–Q9), then DPO cells (D1–D10).
- DPO depends on the QLoRA adapter — cell D4 validates adapter zip structure
  before uploading.
- Includes OOM recovery notes and disconnect recovery notes.

DPO must not be run before QLoRA. The scripts enforce this: `train_phi3_dpo.py`
checks for the adapter at `adapters/phi3_veritas_qlora` and emits a blocked
report if it is missing.

---

## Mac-local MLX LoRA (measured)

The Mac-local explanation adapter is Qwen2.5-1.5B-Instruct-4bit, not Phi-3.
It is at `checkpoints/mlx_lora_verifier` and `checkpoints/mlx_lora_verifier_300`.

Measured throughput: 53.7 tok/s on M-series Mac (MLX).
Details: `reports/mlx_lora_training_metrics.md`, `reports/mac_local_inference_benchmark.md`.

---

## What improved

- Relevance gate reduces NEI false-positive rate significantly (0.7098 → 0.2857)
  without any additional training.
- Gate is wired up and config-driven, ready to enable if calibrated at claim level.

## What regressed or stayed blocked

- Verifier robustness retrain: regressed on both oracle and retrieved metrics.
- Relevance gate: macro_F1 regressed (net negative until better calibration).
- Phi-3 QLoRA/DPO: blocked on CUDA; datasets and scripts are ready for Colab.

## Production checkpoint

`checkpoints/transformer_verifier_clean` — unchanged, remains baseline.
