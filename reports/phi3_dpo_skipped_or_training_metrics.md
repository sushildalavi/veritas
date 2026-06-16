# Phi-3 DPO Training Report

- status: skipped
- reason: CUDA is unavailable on this machine.
- timestamp_utc: 2026-06-16T00:30:19.701145+00:00
- git_commit: 0a7af7773132d287a4979c967e96d1bbc48524ad
- base_model: microsoft/Phi-3-mini-4k-instruct
- adapter_dir: adapters/phi3_veritas_qlora
- preference_file: data/explanations/dpo_train.jsonl
- eval_file: data/explanations/dpo_val.jsonl
- output_dir: adapters/phi3_veritas_dpo

## Environment

- reason: CUDA is unavailable on this machine.

## Colab / Kaggle Commands

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_qlora.py --config configs/phi3_qlora.yaml
python3 scripts/train_phi3_dpo.py --config configs/phi3_dpo.yaml
```
