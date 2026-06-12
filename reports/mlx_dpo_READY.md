# MLX DPO Alignment — NOT AVAILABLE

## Status: NOT COMPLETE

DPO (Direct Preference Optimization) alignment on top of the new
[MLX LoRA adapter](mlx_lora_READY.md) (`checkpoints/mlx_lora_verifier/`) was
not implemented or run.

## Why

`mlx-lm` 0.31.3 (the installed version on this machine) only supports
supervised fine-tuning via `mlx_lm.lora` with
`--fine-tune-type {lora, dora, full}`. There is no `dpo` fine-tune type, no
`mlx_lm.tuner` DPO trainer module, and no separate `mlx-lm-dpo` package
installed (`find mlx_lm -iname '*dpo*'` returns nothing).

Implementing DPO from scratch (custom preference-loss training loop,
reference-model logprob computation, preference dataset construction) would
be a significant new component well beyond "if MLX DPO support is available
and simple." Per project rules, no `scripts/train_mlx_dpo.py`,
`configs/mlx_dpo.yaml`, adapter, or `reports/mlx_dpo_eval.*` have been
created or fabricated.

## What exists today

- The CUDA-path DPO blocker remains documented in
  [`reports/dpo_BLOCKED_QLORA_REQUIRED.md`](dpo_BLOCKED_QLORA_REQUIRED.md)
  (depends on the CUDA QLoRA adapter, which also does not exist).
- The MLX LoRA adapter (`checkpoints/mlx_lora_verifier/`) is a valid SFT
  starting point if DPO support is added to `mlx-lm` in the future, or if a
  custom MLX DPO loop is implemented as a separate, explicitly-scoped task.

## How to unblock

1. Check for a newer `mlx-lm` release with native DPO/preference-tuning
   support (`pip show mlx-lm`; check `mlx_lm.tuner` for a DPO trainer).
2. If still unsupported, implement a custom MLX preference-training loop
   (reference model + policy model logprob ratio loss) as its own scoped
   task, starting from `checkpoints/mlx_lora_verifier/`.

## Required summary line

**MLX DPO: NOT COMPLETE — no DPO trainer available in installed `mlx-lm`
(0.31.3); not implemented from scratch per project rules.**
