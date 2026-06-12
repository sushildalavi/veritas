# Final Faithfulness Comparison

- Verifier checkpoint: `checkpoints/transformer_verifier_clean`
- Data file: `data/processed/verifier_val.jsonl`

| generator | status | example_count | verifier_accuracy | citation_valid_rate | mean_citation_precision | mean_unsupported_sentence_rate | verdict_consistency_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| template | measured | 200 | 0.575 | 0.560 | 1.000 | 0.306 | 0.755 |
| qlora | not trained - GPU required | - | - | - | - | - | - |
| dpo | not trained - QLoRA required | - | - | - | - | - | - |

## Notes

- **template**: rag.generate_template_explanation driven by checkpoints/transformer_verifier_clean predictions (measured)
- **qlora**: TinyLlama + QLoRA explanation generator (see reports/qlora_BLOCKED_GPU_REQUIRED.md) (not trained - GPU required)
- **dpo**: DPO-aligned explanation generator (see reports/dpo_BLOCKED_QLORA_REQUIRED.md) (not trained - QLoRA required)

