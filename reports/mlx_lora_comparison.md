# MLX LoRA Adapter Comparison

| Adapter | Iters | Sample size | Verdict acc | Macro F1 | Citation valid | Unsupported rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoints/mlx_lora_verifier | 100 | 200 | 0.695 | 0.4632 | 0.6 | 0.2537 |
| checkpoints/mlx_lora_verifier_300 | 300 | 200 | 0.52 | 0.24 | 0.545 | 0.3021 |

**Best adapter: `checkpoints/mlx_lora_verifier`** (`reports/mlx_lora_eval_200.json`)

## 600-iteration run: not attempted

The 300-iteration adapter (`checkpoints/mlx_lora_verifier_300`) clearly
overfit relative to the 100-iteration adapter on the same 200-example
validation sample:

- Verdict accuracy dropped from 0.695 to 0.52.
- Macro F1 dropped from 0.4632 to 0.24.
- REFUTED per-class F1 collapsed from 0.6872 to 0.04 (the 300-iter adapter
  over-predicts SUPPORTED).
- Validation loss kept decreasing (0.727 -> 0.440), so loss alone is
  misleading here.

Per the stop condition ("if 300/600 overfits or degrades, keep the 100-iters
adapter and document"), training a 600-iteration adapter was skipped — it
would only extend the same overfitting trend on this small (600-example)
training set. `checkpoints/mlx_lora_verifier` (100 iterations) remains the
adapter referenced from `reports/mlx_lora_READY.md` and the README.
