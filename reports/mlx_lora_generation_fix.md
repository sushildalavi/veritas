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

The explanation adapter was retrained first at 80 iters, then extended to 300 iters
with the correct base model:

- Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Adapter path: `adapters/mlx_qwen_veritas_lora`
- Training iters: 300
- Best val loss: 0.427 (iter 200)
- Final val loss: 0.622 (iter 300, slight overfit)

## Generation After Fix

Generation succeeds — no matmul errors. Evaluated at 300 iters on 10 examples:

| metric | base model | adapter (300 iters) |
|---|---|---|
| format_correctness | 0.0 | **0.2** |
| citation_presence | 0.0 | 0.1 |
| decision_label_consistency | 0.0 | 0.1 |
| avg explanation length (words) | 0.0 | 34.2 |

The base model does not follow the SFT completion format at all. The adapter
achieves partial format compliance (20%) with 300 iters / 256 examples. Some
outputs correctly emit "Decision: / Explanation: / Citations:" structure; others
do not. Prompt leakage after `<|endoftext|>` is visible in some samples — adding
a stop-token or `max_tokens` cap would suppress it.

Source: `reports/explanation_model_eval.json`

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
