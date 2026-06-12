# MLX LoRA Verifier Evaluation

- base_model: mlx-community/Qwen2.5-1.5B-Instruct-4bit
- adapter_path: checkpoints/mlx_lora_verifier
- fine_tune_type: lora
- lora_iters: 100
- train_data: data/processed/mlx_lora
- eval_file: data/processed/verifier_val.jsonl
- sample_size: 20

| Metric | Value |
| --- | ---: |
| Verdict accuracy | 0.6 |
| Macro F1 | 0.4 |
| Citation valid rate | 0.5 |
| Unsupported sentence rate | 0.3017 |
| Mean latency (s/example) | 1.5346 |
| Parseable verdicts | 20 / 20 |
