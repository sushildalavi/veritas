# Training Artifacts

This repository now separates verdict classification from explanation training.

## Boundaries

- The production verifier remains unchanged.
- The relevance gate stays disabled by default.
- Explanation tuning is for faithfulness and formatting, not automatic verifier improvement.
- Mac-local MLX LoRA is not the same as Phi-3 QLoRA.
- Phi-3 QLoRA and Phi-3 DPO require CUDA.

## Artifacts

| Component | Status | Artifact | Notes |
| --- | --- | --- | --- |
| Retrieved verifier robustness retrain | negative result | `reports/transformer_verifier_robust_eval.json` | Regression documented; production verifier unchanged |
| Relevance gate | negative result | `reports/oracle_vs_retrieved_v2_full_gated.json` | Disabled by default |
| SFT explanation dataset | built | `data/explanations/sft_{train,val,test}.jsonl` | Grounded explanation tuning data |
| MLX LoRA verdict-prediction adapter | trained | `checkpoints/mlx_lora_verifier` | Qwen2.5-1.5B-Instruct-4bit; 0.695 acc, 0.4632 macro-F1 on 200-example eval |
| MLX LoRA explanation adapter | generation bug fixed | `adapters/mlx_qwen_veritas_lora/` | Was trained on wrong base model (Qwen/Qwen3-0.6b); retrained on Qwen2.5-1.5B-Instruct-4bit after script fix; see `reports/mlx_lora_generation_fix.md` |
| Phi-3 QLoRA | skipped | `reports/phi3_qlora_skipped_or_training_metrics.json` | CUDA path only |
| DPO preferences | built | `data/explanations/dpo_{train,val}.jsonl` | Synthetic rejection pairs documented |
| Phi-3 DPO | skipped | `reports/phi3_dpo_skipped_or_training_metrics.json` | Depends on CUDA and the QLoRA path |

## Usage Notes

- Use `scripts/build_explanation_sft_dataset.py` to regenerate the SFT explanation corpus.
- Use `scripts/build_dpo_preferences.py` to regenerate preference pairs.
- Use `scripts/train_phi3_qlora.py --dry-run --config configs/phi3_qlora.yaml` to validate the QLoRA path before Colab.
- Use `scripts/train_phi3_dpo.py --dry-run --config configs/phi3_dpo.yaml` to validate the DPO path before Colab.
- Use `scripts/train_mlx_lora_explanations.py` for a local Mac MLX LoRA run.
- Use `scripts/train_phi3_qlora.py` and `scripts/train_phi3_dpo.py` only on CUDA hardware.
- Use `scripts/eval_explanation_model.py` to compare explanation quality across backends.

## What Not To Claim

- Do not claim verifier classification improvement from explanation tuning unless a verifier evaluation measures it.
- Do not call the MLX adapter Phi-3 QLoRA.
- Do not describe preference reranking as DPO.
- Do not claim a real Phi-3 checkpoint unless the adapter directory exists.
- Do not claim the robust verifier improved anything — it regressed on both oracle and retrieved metrics.
- Do not claim the relevance gate improved macro-F1 — it improved NEI FPR but regressed macro-F1.
- Do not claim ONNX is faster than native transformers on this Mac — it is not (55 vs 62 ex/s at batch=1).
