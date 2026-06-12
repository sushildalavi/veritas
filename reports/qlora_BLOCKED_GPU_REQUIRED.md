# QLoRA Fine-Tuning — BLOCKED (GPU Required)

## Status: NOT COMPLETE

QLoRA fine-tuning of `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for citation-grounded
explanation generation could not be executed in this environment.

## Why

- `torch.cuda.is_available()` is `False` on this machine (Apple Silicon, MPS
  only — no CUDA device).
- `bitsandbytes` (required for 4-bit NF4 quantization) does not support
  this platform and is not installed.

QLoRA as specified (4-bit quantized base model + LoRA adapters via
`bitsandbytes`/`peft`) requires a CUDA GPU. There is no CPU/MPS fallback that
would produce a faithful result, so per project rules no metrics, adapter
files, or checkpoints have been fabricated.

## What has been prepared (ready to run on a CUDA machine)

- [`configs/qlora_tinyllama.yaml`](../configs/qlora_tinyllama.yaml) — full
  training configuration (base model, quantization, LoRA, training
  hyperparameters, report paths).
- [`scripts/train_qlora_real.py`](../scripts/train_qlora_real.py) — complete,
  ready-to-run training script that:
  - Verifies CUDA + `bitsandbytes` availability before doing any work.
  - Builds an explanation-generation dataset from
    `data/processed/verifier_train.jsonl` / `verifier_val.jsonl` using
    `rag.context_builder.build_context` and
    `rag.prompt_templates.EXPLANATION_PROMPT`, with template explanations
    (`rag.generate_template_explanation`) as SFT targets.
  - Loads the base model in 4-bit NF4 via `BitsAndBytesConfig`, applies
    `prepare_model_for_kbit_training` + a LoRA adapter
    (`r=16, alpha=32, dropout=0.05`, targeting `q_proj/k_proj/v_proj/o_proj`).
  - Trains via `transformers.Trainer` and saves the adapter to
    `checkpoints/qlora_tinyllama/`.
  - Evaluates citation faithfulness on the validation split via
    `rag.check_citations` (citation validity rate, mean citation precision,
    verdict consistency rate) and writes `reports/qlora_eval.{json,md}`.

## How to run (on a CUDA machine, e.g. Colab/Kaggle T4 or better)

```bash
pip install torch transformers peft bitsandbytes datasets accelerate
python scripts/train_qlora_real.py --config configs/qlora_tinyllama.yaml \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

(The script also accepts individual `--base-model`, `--output-dir`,
`--max-train-examples`, etc. flags; see `--help`.)

## Required user message

**PROJECT NOT COMPLETE: GPU REQUIRED FOR QLORA.**

Per project rules, QLoRA is not marked complete because no adapter files or
`reports/qlora_eval.*` exist from a real training run. DPO (Phase 7) depends
on QLoRA and is therefore also blocked — see
[`reports/dpo_BLOCKED_QLORA_REQUIRED.md`](dpo_BLOCKED_QLORA_REQUIRED.md).
