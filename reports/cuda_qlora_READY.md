# CUDA QLoRA Fine-Tuning — READY, NOT RUN

## Status: NOT COMPLETE (READY_NOT_RUN)

CUDA QLoRA fine-tuning of `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on the
Verdict/Explanation/Citation verification task has not been executed. This
machine (Apple Silicon, MPS only) cannot run it: `torch.cuda.is_available()`
is `False` and `bitsandbytes` does not support this platform.

This is the CUDA counterpart to the completed
[MLX LoRA adapter](mlx_lora_READY.md) (`checkpoints/mlx_lora_verifier/`), and
is a prerequisite for CUDA DPO (`reports/cuda_dpo_READY.md`).

## What has been prepared (ready to run on a CUDA machine)

- [`configs/cuda_qlora_tinyllama.yaml`](../configs/cuda_qlora_tinyllama.yaml) —
  base model, quantization, LoRA, and training hyperparameters.
- [`scripts/train_cuda_qlora.py`](../scripts/train_cuda_qlora.py) — complete,
  ready-to-run training script that:
  - Loads `data/processed/mlx_lora/{train,valid}.jsonl` (600/100 examples,
    the same data used for MLX LoRA) and flattens each `{"messages": [...]}`
    chat example into `{"prompt", "completion"}` pairs, written to
    `data/processed/sft_train.jsonl` / `sft_val.jsonl`
    (`--export-sft-only`, CPU-only — already run, see below).
  - Verifies CUDA + `bitsandbytes` availability before doing any GPU work;
    if unavailable, writes `reports/cuda_qlora_eval_FAILED.md` and exits
    non-zero instead of fabricating metrics.
  - Loads the base model in 4-bit NF4 via `BitsAndBytesConfig`, applies
    `prepare_model_for_kbit_training` + a LoRA adapter
    (`r=16, alpha=32, dropout=0.05`, targeting `q_proj/k_proj/v_proj/o_proj`).
  - Trains via `transformers.Trainer` and saves the adapter to
    `checkpoints/cuda_qlora_verifier/`.
  - Evaluates the adapter on `data/processed/mlx_lora/valid.jsonl` (100
    examples) using the **same metrics as the MLX LoRA evaluation**
    (`evaluation/mlx_lora_eval.py`): verdict accuracy, macro F1, per-class
    F1, citation valid rate, unsupported-sentence rate, mean latency, and
    writes `reports/cuda_qlora_eval.{json,md}`.
- [`notebooks/12_cuda_qlora_kaggle_colab.ipynb`](../notebooks/12_cuda_qlora_kaggle_colab.ipynb)
  — installs dependencies, checks for a CUDA GPU, clones the repo, exports
  the SFT dataset, runs training + evaluation, zips the adapter/reports/SFT
  files, and lists exactly which files to copy back into this repo.

## Already verified on this machine (CPU-only, no fabricated metrics)

`python3 scripts/train_cuda_qlora.py --export-sft-only` was run locally and
produced `data/processed/sft_train.jsonl` (600 examples) and
`data/processed/sft_val.jsonl` (100 examples) — each
`{"prompt": "<system>\\n\\n<user>", "completion": "Verdict: ...\\nExplanation: ...\\nCitation: ..."}`,
matching the prompt format used by `data/processed/preference_pairs.jsonl`
(Phase 3) so the resulting adapter is a valid starting point for CUDA DPO.

## How to run (on a CUDA machine, e.g. Colab/Kaggle T4 or better)

Open [`notebooks/12_cuda_qlora_kaggle_colab.ipynb`](../notebooks/12_cuda_qlora_kaggle_colab.ipynb)
in Kaggle or Colab with a GPU runtime and run all cells, or from a CUDA shell:

```bash
pip install torch transformers peft bitsandbytes accelerate datasets trl
python3 scripts/train_cuda_qlora.py \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output-dir checkpoints/cuda_qlora_verifier \
    --report-json reports/cuda_qlora_eval.json \
    --report-md reports/cuda_qlora_eval.md
```

## Required summary line

**CUDA QLoRA: READY, NOT RUN** — `configs/cuda_qlora_tinyllama.yaml`,
`scripts/train_cuda_qlora.py`, and
`notebooks/12_cuda_qlora_kaggle_colab.ipynb` are complete and the SFT dataset
export (`data/processed/sft_train.jsonl` / `sft_val.jsonl`) has been
verified locally, but no adapter or `reports/cuda_qlora_eval.*` exist because
this machine has no CUDA GPU. Do not claim CUDA QLoRA complete until
`checkpoints/cuda_qlora_verifier/` and `reports/cuda_qlora_eval.json` exist
from an actual run.
