# Phi-3 QLoRA Training Report

- status: skipped
- reason: CUDA is unavailable on this machine.
- timestamp_utc: 2026-06-16T00:30:19.701174+00:00
- git_commit: 0a7af7773132d287a4979c967e96d1bbc48524ad
- base_model: microsoft/Phi-3-mini-4k-instruct
- train_file: data/explanations/sft_train.jsonl
- eval_file: data/explanations/sft_val.jsonl
- adapter_path: adapters/phi3_veritas_qlora

## Environment

- reason: CUDA is unavailable on this machine.

## Colab / Kaggle Commands

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_qlora.py --config configs/phi3_qlora.yaml
```
