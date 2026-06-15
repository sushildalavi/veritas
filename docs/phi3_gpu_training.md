# Phi-3 GPU Training

This repository now contains a Phi-3-mini QLoRA and DPO path, but those runs were not executed on this CPU-only machine.

## QLoRA

Build the dataset:

```bash
python3 scripts/build_phi3_qlora_dataset.py
```

Train on Colab or Kaggle T4:

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_qlora.py --config configs/phi3_qlora.yaml
```

Expected outputs if training succeeds:

- `adapters/phi3_veritas_qlora/`
- `reports/qlora_training_metrics.json`
- `reports/qlora_before_after_examples.md`

## DPO

Build preferences:

```bash
python3 scripts/build_phi3_dpo_preferences.py
```

Train after QLoRA exists:

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_dpo.py --config configs/phi3_dpo.yaml
```

Expected outputs if training succeeds:

- `data/dpo_preferences/preferences.jsonl`
- `adapters/phi3_veritas_dpo/`
- `reports/dpo_training_metrics.json`
- `reports/dpo_preference_eval.json`

## Environment notes

- These scripts intentionally emit blocked reports instead of fake checkpoints when CUDA or adapter prerequisites are missing.
- vLLM remains explanation-serving only; it does not replace verifier labels.
