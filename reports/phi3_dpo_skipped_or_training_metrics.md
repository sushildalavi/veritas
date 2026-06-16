# Phi-3 DPO Training Report

- status: dry_run
- reason: 
- timestamp_utc: 2026-06-16T05:51:25.168728+00:00
- git_commit: b82e60418fc0c1928eb4e5646fc59da85c07f9a3
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

## Colab / Kaggle Commands

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_qlora.py --config configs/phi3_qlora.yaml
python3 scripts/train_phi3_dpo.py --config configs/phi3_dpo.yaml
```

## Run Notes

- This is a readiness check only.
- No training was executed.
- The report confirms local file paths and command syntax.
