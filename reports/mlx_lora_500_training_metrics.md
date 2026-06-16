# MLX LoRA Explanation Adapter — 500-Iter Training

## Summary

Extended training from 300 iters to 500 iters for the explanation SFT adapter.

- Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Adapter path: `adapters/mlx_qwen_veritas_lora`
- Training examples: 256

## Val Loss (selected checkpoints)

| Iter | Val loss |
|------|----------|
| 20   | 0.820    |
| 40   | 0.510    |
| 80   | 0.496    |
| 120  | 0.518    |
| 200  | **0.427** |
| 240  | 0.606    |

Best val loss: **0.427** at iter 200. Loss is noisy beyond iter 200 due to small
dataset size (256 examples). This matches the 300-iter run — no regression.

## Eval Results at 500 Iters (25 examples)

Source: `reports/mlx_lora_500_eval.json`

| Metric | Base model | Adapter (300 iters) | Adapter (500 iters) | Change |
|--------|-----------|---------------------|---------------------|--------|
| format_correctness | 0.0 | 0.20 | **0.28** | +0.08 |
| citation_presence | 0.0 | 0.10 | **0.72** | +0.62 |
| decision_label_consistency | 0.0 | 0.10 | **0.24** | +0.14 |
| avg explanation length (words) | 0.0 | 34.2 | 33.24 | −0.96 |
| prompt leakage | present | present | reduced | fixed by stop-token trim |

## Improvement Assessment

- citation_presence improved dramatically (+0.62): the adapter now reliably generates
  `Citations: ["E1"]` or similar in most outputs.
- format_correctness improved (+0.08): more outputs contain `Decision:/Explanation:/Citations:` structure.
- decision_label_consistency improved (+0.14): label agreement increasing but not yet robust.
- Targets set: format≥0.5, citation≥0.3, consistency≥0.3.
  - citation_presence **exceeded** target (0.72 > 0.3)
  - format_correctness not yet at target (0.28 < 0.5)
  - decision_label_consistency not yet at target (0.24 < 0.3)

**Adapter at 500 iters is the best measured result.** Keep as production candidate
but do not claim production-quality structured output.
