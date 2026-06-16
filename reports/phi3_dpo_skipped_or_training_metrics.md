# Phi-3 DPO Training Report

- status: dry_run
- reason: 
- timestamp_utc: 2026-06-16T01:46:47.237929+00:00
- git_commit: 3f9bd11514c3b04d132d95db2f7d8a98f05e136e
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
