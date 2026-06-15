# QLoRA Blocked

Phi-3-mini QLoRA was not trained in this environment.

- Base model: `microsoft/Phi-3-mini-4k-instruct`
- Train file: `data/processed/phi3_qlora/train.jsonl`
- Expected output dir: `adapters/phi3_veritas_qlora`

Reason: CUDA GPU with bitsandbytes/PEFT support is required.

Use the Colab/Kaggle commands in `docs/phi3_gpu_training.md`.

No checkpoint or training metrics were fabricated.