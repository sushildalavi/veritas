# Veritas

Veritas | LLM Fact Verification with Hybrid Retrieval, Learned Ranking, QLoRA, and DPO Alignment

Focused goal: given a claim, retrieve evidence, rank passages, verify the claim as `SUPPORTED` / `REFUTED` / `NOT ENOUGH INFO`, and produce grounded explanations with citations.

## Architecture

_Placeholder for the full system diagram._

## Results

| Component | Metric | Value |
| --- | --- | --- |
| Retrieval | Recall@k | TODO |
| Ranking | nDCG@10 | TODO |
| Verification | Macro-F1 | TODO |
| Faithfulness | Citation precision | TODO |

## Deployment

The public demo is designed for free hosting on Hugging Face Spaces with Gradio and must not require a hosted 7B model.

## Notes

- No fake metrics are reported in this repository.
- Values in tables and claims in the README remain `TODO` until real experiments are run.
