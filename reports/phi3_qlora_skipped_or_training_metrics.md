# Phi-3 QLoRA Training Report

- status: skipped
- reason: CUDA is unavailable on this machine.
- timestamp_utc: 2026-06-16T05:51:32.211950+00:00
- git_commit: b82e60418fc0c1928eb4e5646fc59da85c07f9a3
- base_model: microsoft/Phi-3-mini-4k-instruct
- train_file: data/explanations/sft_train.jsonl
- eval_file: data/explanations/sft_val.jsonl
- adapter_path: adapters/phi3_veritas_qlora

## Preflight Checks

| Check | Path | Exists |
| --- | --- | ---: |
| train_file_exists | data/explanations/sft_train.jsonl | true |
| eval_file_exists | data/explanations/sft_val.jsonl | true |
| output_parent_exists | adapters | true |
| base_model | microsoft/Phi-3-mini-4k-instruct | true |

## Environment

- reason: CUDA is unavailable on this machine.

## Colab / Kaggle Commands

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_qlora.py --config configs/phi3_qlora.yaml
```
