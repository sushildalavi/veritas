# MLX LoRA Generation Fix

## Status: Fixed

Generation now succeeds. The matmul shape mismatch error is resolved.

## Root Cause

`scripts/train_mlx_lora_explanations.py` maps config keys to mlx_lm's `lora_args`
by checking `if key in lora_args`. The YAML config uses `base_model` but mlx_lm's
training parser uses `model` — so the base model was never passed. The script then
applied `mlx_lm.lora.CONFIG_DEFAULTS`, which sets `model = "Qwen/Qwen3-0.6b"`.

The adapter was trained against Qwen3-0.6b (hidden_dim=1024). The evaluation script
loaded the adapter against Qwen2.5-1.5B-Instruct-4bit (hidden_dim=1536), causing a
matmul shape mismatch on every generation call:

```
[matmul] Last dimension of first input with shape (1,N,1536) must match
         second to last dimension of second input with shape (1024,8).
```

## Fix

Added one explicit mapping in `scripts/train_mlx_lora_explanations.py`:

```python
lora_args["model"] = config["base_model"]  # config uses base_model; mlx_lm uses model
```

Placed before the `CONFIG_DEFAULTS` loop so the explicit value takes precedence.

## Retrain

The explanation adapter was retrained at 80 iters with the correct base model:

- Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Adapter path: `adapters/mlx_qwen_veritas_lora`
- Train loss at iter 80: 0.562
- Val loss at iter 80: 0.496

## Generation After Fix

Generation succeeds — no matmul errors. Format compliance is low at this training
scale (80 iters / 256 examples). The model generates text but does not reliably
follow the "Decision: / Explanation: / Citations:" structured format. More training
iterations would improve format compliance.

## Verdict Prediction Adapter (properly trained)

`checkpoints/mlx_lora_verifier` was trained correctly (100 iters,
`mlx-community/Qwen2.5-1.5B-Instruct-4bit`, verifier-format data). Its measured
results on 200 examples:

| metric | value |
|---|---|
| verdict_accuracy | 0.695 |
| macro_f1 | 0.4632 |
| citation_valid_rate | 0.6 |
| parseable_rate | 1.0 |

This adapter is for verdict prediction, not explanation SFT. See
`reports/mlx_lora_eval_200.json`.
