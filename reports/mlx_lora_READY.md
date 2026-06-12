# MLX LoRA Fine-Tuning — Apple Silicon (COMPLETE)

## Status: COMPLETE (local, no CUDA, no bitsandbytes)

This is the Apple Silicon alignment path that replaces the CUDA/bitsandbytes
QLoRA plan (see [`reports/qlora_BLOCKED_GPU_REQUIRED.md`](qlora_BLOCKED_GPU_REQUIRED.md),
which remains the CUDA cloud-only fallback). It uses
[`mlx-lm`](https://github.com/ml-explore/mlx-examples)'s `mlx_lm.lora` LoRA
trainer on an MLX 4-bit-quantized Qwen2.5 model. This is **MLX LoRA**, not
QLoRA — the base model ships pre-quantized by `mlx-community`, and the
adapter itself is full-precision LoRA (not quantized).

## What was run

- Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (already cached
  locally, 839MB).
- Fine-tune type: `lora` (`fine_tune_type: lora` in
  [`configs/mlx_lora_qwen05b.yaml`](../configs/mlx_lora_qwen05b.yaml)),
  8 trainable layers, LoRA rank 8 (mlx-lm defaults).
- Training data: 600 examples from
  [`data/processed/mlx_lora/train.jsonl`](../data/processed/mlx_lora/train.jsonl)
  (converted from `data/processed/verifier_train.jsonl` by
  [`scripts/build_mlx_lora_dataset.py`](../scripts/build_mlx_lora_dataset.py)),
  100 validation / 100 test examples.
- Training: 100 iterations, batch size 1, learning rate 1e-5, max sequence
  length 512, gradient checkpointing on. Train loss 1.535 -> 0.508, val loss
  2.254 -> 0.727.
- Adapter saved to
  [`checkpoints/mlx_lora_verifier/adapters.safetensors`](../checkpoints/mlx_lora_verifier/adapters.safetensors)
  (2.638M trainable params, 0.171% of 1.544B).

## How to run

```bash
make build-mlx-lora-data   # scripts/build_mlx_lora_dataset.py
make train-mlx-lora         # scripts/train_mlx_lora.py (trains + evaluates)
make eval-mlx-lora          # scripts/train_mlx_lora.py --skip-train (re-eval only)
```

Or directly:

```bash
python3 scripts/build_mlx_lora_dataset.py
python3 scripts/train_mlx_lora.py --config configs/mlx_lora_qwen05b.yaml
```

## Results

See [`reports/mlx_lora_eval.md`](mlx_lora_eval.md) /
[`reports/mlx_lora_eval.json`](mlx_lora_eval.json) for the full evaluation
(20 held-out validation examples): verdict accuracy 0.6, macro F1 0.4,
citation valid rate 0.5, unsupported-sentence rate 0.30, mean latency
1.53s/example.

## What can be claimed

- A real LoRA adapter was trained on Apple Silicon (MLX, no CUDA, no
  bitsandbytes) and is checked in at `checkpoints/mlx_lora_verifier/`.
- The adapter produces parseable `Verdict: / Explanation: / Citation:`
  output on 20/20 held-out examples, with measured (not fabricated) accuracy
  and citation-faithfulness metrics in `reports/mlx_lora_eval.{json,md}`.

## What cannot be claimed

- This is a small run (100 iterations, 600 training examples, 20-example
  eval) — not a production-scale fine-tune. Metrics should be read as a
  small-sample signal, not a benchmark result.
- This is **not** quantized LoRA (no QLoRA-style 4-bit adapter training) —
  the base model weights are pre-quantized by `mlx-community`, but the LoRA
  adapter is trained at full precision via `mlx-lm`.
- DPO alignment was not run on top of this adapter — see
  [`reports/mlx_dpo_READY.md`](mlx_dpo_READY.md) for why.
- CUDA/bitsandbytes QLoRA (`scripts/train_qlora_real.py`,
  `configs/qlora_tinyllama.yaml`) remains an optional cloud-only path and is
  still not run — see
  [`reports/qlora_BLOCKED_GPU_REQUIRED.md`](qlora_BLOCKED_GPU_REQUIRED.md).
