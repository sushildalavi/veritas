# DPO Alignment — BLOCKED (QLoRA Required)

## Status: NOT COMPLETE

Direct Preference Optimization (DPO) fine-tuning was not attempted because it
depends on the QLoRA adapter produced in Phase 6, which is itself blocked —
see [`reports/qlora_BLOCKED_GPU_REQUIRED.md`](qlora_BLOCKED_GPU_REQUIRED.md).

## Why

Per project rules, DPO must start from the QLoRA-fine-tuned model
(`checkpoints/qlora_tinyllama/`). That adapter does not exist because this
machine has no CUDA GPU and `bitsandbytes` is unavailable. Running DPO from
the unmodified base model instead would not match the intended pipeline and
would risk producing misleading "DPO" results, so no DPO adapter, training
run, or `reports/dpo_eval.*` has been created or fabricated.

## What exists today

- `models/dpo_model.py` — `DPOModelConfig` dataclass and `load_dpo_model()`
  stub (`{"available": False}`).
- `training/train_dpo.py` — stub that attempts to build a `trl.DPOTrainer`.
- `configs/dpo.yaml` — existing config (phi-2-based, marked "offline optional
  extension only").

## How to unblock

1. Run `scripts/train_qlora_real.py` on a CUDA machine (Colab/Kaggle T4 or
   better) per `reports/qlora_BLOCKED_GPU_REQUIRED.md` to produce
   `checkpoints/qlora_tinyllama/` and `reports/qlora_eval.{json,md}`.
2. Build a preference dataset (chosen vs. rejected explanations, e.g. citation
   -faithful template explanation vs. an unsupported/uncited generation).
3. Write a real `scripts/train_dpo_real.py` (analogous to
   `scripts/train_qlora_real.py`) that loads the QLoRA adapter, runs
   `trl.DPOTrainer`, saves a DPO adapter to `checkpoints/dpo_tinyllama/`, and
   writes `reports/dpo_eval.{json,md}`.

## Required user message

**PROJECT NOT COMPLETE: QLORA REQUIRED FOR DPO.**
