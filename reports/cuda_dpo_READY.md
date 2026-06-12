# CUDA DPO Alignment — READY, NOT RUN

## Status: NOT COMPLETE (READY_NOT_RUN)

Direct Preference Optimization (DPO) alignment of the CUDA QLoRA verifier
adapter has not been executed. It depends on
[`checkpoints/cuda_qlora_verifier/`](cuda_qlora_READY.md), which also does not
exist on this machine (no CUDA GPU). Per project rules, no DPO adapter,
training run, or `reports/cuda_dpo_eval.*` has been created or fabricated.

This supersedes the older, now-stale
[`reports/dpo_BLOCKED_QLORA_REQUIRED.md`](dpo_BLOCKED_QLORA_REQUIRED.md)
(which referenced the unused `checkpoints/qlora_tinyllama/` / `configs/dpo.yaml`
path). The current path is CUDA QLoRA (Phase 4) → CUDA DPO (this phase),
both via Kaggle/Colab notebooks.

The MLX-only equivalent remains documented separately in
[`reports/mlx_dpo_READY.md`](mlx_dpo_READY.md) (blocked: `mlx-lm` 0.31.3 has
no DPO trainer).

## What has been prepared (ready to run on a CUDA machine)

- [`data/processed/preference_pairs.jsonl`](../data/processed/preference_pairs.jsonl)
  — 1,382 `{"prompt", "chosen", "rejected"}` pairs (Phase 3, see
  [`reports/preference_pair_stats.md`](preference_pair_stats.md)), where
  `chosen` is a correct verdict with a citation-valid grounded explanation,
  and `rejected` is one of: wrong verdict, invalid citation, or unsupported
  explanation (461/461/460 split).
- [`configs/cuda_dpo_tinyllama.yaml`](../configs/cuda_dpo_tinyllama.yaml) —
  DPO hyperparameters (`beta=0.1`, `max_length=512`,
  `max_prompt_length=384`) and paths.
- [`scripts/train_cuda_dpo.py`](../scripts/train_cuda_dpo.py) — complete,
  ready-to-run script that:
  - Verifies `checkpoints/cuda_qlora_verifier/adapter_config.json` exists and
    that CUDA + `bitsandbytes` + `peft` + `trl` are available; if not, writes
    `reports/cuda_dpo_eval_FAILED.md` and exits non-zero instead of
    fabricating metrics.
  - Loads the CUDA QLoRA adapter as the starting policy
    (`PeftModel.from_pretrained(..., is_trainable=True)`) over a 4-bit
    quantized `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
  - Evaluates this adapter on `data/processed/mlx_lora/valid.jsonl` (100
    examples) **before** DPO, using the same metrics as the MLX/CUDA QLoRA
    evaluations (`evaluation/cuda_verifier_eval.py`): verdict accuracy, macro
    F1, per-class F1, citation valid rate, unsupported-sentence rate, and
    **verdict consistency rate**.
  - Runs `trl.DPOTrainer` on `data/processed/preference_pairs.jsonl`
    (`ref_model=None`, so TRL derives the reference logits from the adapter
    itself), saves the result to `checkpoints/cuda_dpo_verifier/`.
  - Evaluates the DPO-aligned adapter **after** DPO on the same validation
    set, and writes `reports/cuda_dpo_eval.{json,md}` with a side-by-side
    before/after/delta table.
- [`notebooks/13_cuda_dpo_kaggle_colab.ipynb`](../notebooks/13_cuda_dpo_kaggle_colab.ipynb)
  — installs dependencies, checks for a CUDA GPU, clones the repo, loads the
  uploaded CUDA QLoRA adapter from Phase 4, runs DPO training +
  before/after evaluation, zips the adapter/reports, and lists exactly which
  files to copy back into this repo.

## How to run (on a CUDA machine, e.g. Colab/Kaggle T4 or better)

1. First complete CUDA QLoRA (Phase 4,
   [`reports/cuda_qlora_READY.md`](cuda_qlora_READY.md)) and copy
   `checkpoints/cuda_qlora_verifier/` into the repo.
2. Open [`notebooks/13_cuda_dpo_kaggle_colab.ipynb`](../notebooks/13_cuda_dpo_kaggle_colab.ipynb)
   in Kaggle or Colab with a GPU runtime and run all cells, or from a CUDA shell:

```bash
pip install torch transformers peft bitsandbytes accelerate datasets trl
python3 scripts/train_cuda_dpo.py \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --qlora-adapter-path checkpoints/cuda_qlora_verifier \
    --preference-pairs-file data/processed/preference_pairs.jsonl \
    --output-dir checkpoints/cuda_dpo_verifier \
    --report-json reports/cuda_dpo_eval.json \
    --report-md reports/cuda_dpo_eval.md
```

## Required summary line

**CUDA DPO: READY, NOT RUN** — `data/processed/preference_pairs.jsonl`,
`configs/cuda_dpo_tinyllama.yaml`, `scripts/train_cuda_dpo.py`, and
`notebooks/13_cuda_dpo_kaggle_colab.ipynb` are complete, but no DPO adapter or
`reports/cuda_dpo_eval.*` exist because this machine has no CUDA GPU and the
prerequisite CUDA QLoRA adapter does not exist either. Do not claim DPO
complete until `checkpoints/cuda_dpo_verifier/` and
`reports/cuda_dpo_eval.json` exist from an actual run. Until then, only claim
that the preference-pair dataset (Phase 3) was constructed, not that DPO
alignment was performed.
