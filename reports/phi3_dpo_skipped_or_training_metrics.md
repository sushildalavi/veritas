# Phi-3 DPO Training Report

- status: skipped
- reason: CUDA is unavailable on this machine.
- timestamp_utc: 2026-06-16T01:40:26.736390+00:00
- git_commit: 512c986b79b707b785901283dc6842f3a65f378f
- base_model: microsoft/Phi-3-mini-4k-instruct
- adapter_dir: adapters/phi3_veritas_qlora
- preference_file: data/explanations/dpo_train.jsonl
- eval_file: data/explanations/dpo_val.jsonl
- output_dir: adapters/phi3_veritas_dpo

## Preflight Checks

| Check | Path | Exists |
| --- | --- | ---: |
| preference_file_exists | data/explanations/dpo_train.jsonl | true |
| eval_file_exists | data/explanations/dpo_val.jsonl | true |
| output_parent_exists | adapters | true |
| base_model | microsoft/Phi-3-mini-4k-instruct | true |
| adapter_exists | adapters/phi3_veritas_qlora | false |

## Environment

- reason: CUDA is unavailable on this machine.

## Colab / Kaggle Commands

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_qlora.py --config configs/phi3_qlora.yaml
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_dpo.py --config configs/phi3_dpo.yaml
```
