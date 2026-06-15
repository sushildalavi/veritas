# DPO Blocked

Phi-3-mini DPO was not trained in this environment.

- Base model: `microsoft/Phi-3-mini-4k-instruct`
- Preference file: `data/dpo_preferences/preferences.jsonl`
- Expected output dir: `adapters/phi3_veritas_dpo`

Reason: QLoRA adapter checkpoint missing.

Use the Colab/Kaggle commands in `docs/phi3_gpu_training.md`.

No checkpoint or training metrics were fabricated.