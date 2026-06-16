# Phi-3 GPU Training

This is the CUDA-only training path for the Phi-3 family.
The local Mac path is the MLX LoRA adapter documented in `docs/training_artifacts.md`.

## QLoRA

Build the explanation SFT dataset first:

```bash
python3 scripts/build_explanation_sft_dataset.py
```

Train on Colab or Kaggle T4:

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_qlora.py --config configs/phi3_qlora.yaml
```

Expected outputs if training succeeds:

- `adapters/phi3_veritas_qlora/`
- `reports/phi3_qlora_skipped_or_training_metrics.json`
- `reports/phi3_qlora_skipped_or_training_metrics.md`
- `reports/phi3_qlora_before_after_examples.md`

## DPO

Build preference pairs:

```bash
python3 scripts/build_dpo_preferences.py
```

Train after the QLoRA adapter exists:

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
python3 scripts/train_phi3_dpo.py --config configs/phi3_dpo.yaml
```

Expected outputs if training succeeds:

- `adapters/phi3_veritas_dpo/`
- `reports/phi3_dpo_skipped_or_training_metrics.json`
- `reports/phi3_dpo_skipped_or_training_metrics.md`
- `reports/phi3_dpo_before_after_examples.md`

## Environment Notes

- These scripts intentionally emit skipped reports instead of fake checkpoints when CUDA or dependencies are missing.
- QLoRA and DPO are explanation-focused paths and should not be described as verdict classifier improvements unless a verifier evaluation proves that separately.
- The verifier still decides the label; these training runs only affect explanation generation.

