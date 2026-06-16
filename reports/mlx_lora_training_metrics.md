# MLX LoRA Training Metrics

- status: trained
- reason: 
- timestamp_utc: 2026-06-16T06:03:05.620882+00:00
- git_commit: 844c307c591967c7d2b280bdb1eb176a5f989dfd
- base_model: mlx-community/Qwen2.5-1.5B-Instruct-4bit
- dataset_path: /Users/sushildalavi/Desktop/Github/Veritas/data/explanations
- adapter_path: /Users/sushildalavi/Desktop/Github/Veritas/adapters/mlx_qwen_veritas_lora

- trained_examples: 2808
- validation_examples: 649
- test_examples: 642

## Config

```json
{
  "adapter_path": "adapters/mlx_qwen_veritas_lora",
  "base_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
  "batch_size": 1,
  "dataset_dir": "data/explanations",
  "grad_checkpoint": true,
  "iters": 300,
  "learning_rate": 1e-05,
  "max_examples": 256,
  "max_seq_length": 768,
  "num_layers": 8,
  "output_dir": "adapters/mlx_qwen_veritas_lora",
  "save_every": 40,
  "seed": 42,
  "steps_per_eval": 20,
  "steps_per_report": 20,
  "system_prompt": "You are a fact-verification assistant. Use only the provided evidence. Return concise, grounded explanations with citations.",
  "test_file": "sft_test.jsonl",
  "train_file": "sft_train.jsonl",
  "val_batches": 10,
  "val_file": "sft_val.jsonl"
}
```
