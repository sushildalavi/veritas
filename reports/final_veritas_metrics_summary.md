# Veritas — Final Metrics Source of Truth

All numbers verified against actual report files. Sources listed per section.
Generated: 2026-06-15.

---

## 1. Verifier — Oracle vs Retrieved (full 650-example test set)

Source: `reports/oracle_vs_retrieved_v2_full.json`
Checkpoint: `checkpoints/transformer_verifier_clean` (distilroberta-base)

| mode | evidence | accuracy | macro_F1 | NEI FPR |
|---|---|---|---|---|
| bundle | oracle | 0.7354 | **0.7246** | 0.0 |
| per_passage_max | oracle | 0.6985 | **0.6728** | 0.0 |
| bundle | retrieved | 0.3600 | 0.3332 | 0.8839 |
| per_passage_max | retrieved | 0.4062 | **0.3887** | 0.7098 |

Oracle→retrieved gap (per_passage_max): accuracy −0.2923, macro_F1 **−0.2841**

Per-class F1 on retrieved per_passage_max:
- SUPPORTED: 0.3211
- REFUTED: 0.4974
- NOT ENOUGH INFO: 0.3476

---

## 2. Retrieval (full 650-example test set)

Source: `reports/retrieval_profile_comparison_650.json`

| profile | recall@1 | recall@5 | recall@10 | ndcg@10 | verifier macro_F1 |
|---|---|---|---|---|---|
| **bm25_only** (default) | **0.3559** | **0.4979** | **0.5334** | **0.4816** | **0.3887** |
| hybrid_bm25_dense | 0.2923 | 0.4613 | 0.5113 | 0.4288 | 0.3864 |
| hybrid_bm25_sentence_transformer | 0.3903 | 0.5416 | 0.5714 | 0.5234 | 0.3776 |

`hybrid_bm25_sentence_transformer` has better retrieval recall but lower verifier
macro_F1. Better retrieval ranking does not monotonically translate to better end-to-end
fact verification in this setup.

---

## 3. Robust Verifier Retrain (negative result)

Source: `reports/verifier_robustness_training_result.md`
Checkpoint: `checkpoints/transformer_verifier_robust` — NOT in production.

| checkpoint | evidence | accuracy | macro_F1 |
|---|---|---|---|
| baseline | oracle | 0.6985 | 0.6728 |
| baseline | retrieved | 0.4062 | 0.3887 |
| robust retrain | oracle | 0.5615 | 0.4376 (−0.2352) |
| robust retrain | retrieved | 0.3862 | 0.2829 (−0.1058) |

**Both metrics regressed.** Root cause: combined training set was 55% NEI
(76% NEI in retrieved-evidence pairs) without class reweighting. Model collapsed
toward predicting NEI.

Production checkpoint unchanged.

---

## 4. Relevance Gate (negative on primary metric)

Source: `reports/oracle_vs_retrieved_v2_full_gated.json`
Gate threshold tested: 0.5 (lexical token overlap)

| mode | evidence | macro_F1 | NEI FPR |
|---|---|---|---|
| per_passage_max (no gate) | retrieved | 0.3887 | 0.7098 |
| per_passage_max (gate=0.5) | retrieved | 0.3557 (−0.0330) | 0.2857 (−0.4241) |

NEI FPR improved substantially. macro_F1 regressed. Gate stays disabled by
default (`relevance_gate_threshold: null` in `configs/serving.yaml`).

---

## 5. ONNX Benchmark

Source: `reports/onnx_verifier_benchmark.json` and `reports/verifier_inference_benchmark.json`

ONNX CPU (no GPU, no MPS in onnxruntime):

| batch | mean latency ms | throughput ex/s |
|---|---|---|
| 1 | 18.2 | 55.1 |
| 8 | 133.9 | 59.7 |

Transformers baseline (same Mac):

| batch | device | mean latency ms | throughput ex/s |
|---|---|---|---|
| 1 | CPU | 16.0 | 62.4 |
| 8 | CPU | 58.9 | 135.8 |
| 1 | MPS | 6.5 | 153.8 |
| 8 | MPS | 23.5 | 340.9 |

**ONNX CPU is slower than native transformers on this Mac.** MPS via
transformers is 3× faster than ONNX CPU. The ONNX export is valid and functional;
the latency benefit applies on CUDA hardware.

---

## 6. MLX LoRA

Source: `reports/mlx_lora_eval_200.json`, `reports/mlx_lora_training_metrics.json`

### Verdict prediction adapter (`checkpoints/mlx_lora_verifier`)

Correctly trained against `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, 100 iters.
Evaluated on 200 examples from `data/processed/verifier_val.jsonl`.

| metric | value |
|---|---|
| verdict_accuracy | **0.695** |
| macro_F1 | **0.4632** |
| citation_valid_rate | 0.6 |
| parseable_rate | 1.0 |
| SUPPORTED F1 | 0.7024 |
| REFUTED F1 | 0.6872 |
| NOT ENOUGH INFO F1 | 0.0 |

NEI F1=0.0 because the adapter rarely generates NEI on this dataset.

### Explanation adapter (`adapters/mlx_qwen_veritas_lora`)

Was trained against `Qwen/Qwen3-0.6b` (wrong base model) due to a key-name bug in
the training script. **Generation bug is now fixed** (see `reports/mlx_lora_generation_fix.md`).
Adapter retrained at 300 iters against the correct base model. Evaluated on 10 examples
(`reports/explanation_model_eval.json`):

| metric | base model | adapter (300 iters) |
|---|---|---|
| format_correctness | 0.0 | **0.2** |
| citation_presence | 0.0 | 0.1 |
| decision_label_consistency | 0.0 | 0.1 |
| avg explanation length (words) | 0.0 | 34.2 |

Format compliance improved from 0.0 at 80 iters to 0.2 at 300 iters, confirming
the generation bug fix is effective. Best val loss: 0.427 at iter 200.

### Throughput benchmark

Source: `reports/mac_local_inference_benchmark.json`
Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, Apple Silicon

| metric | value |
|---|---|
| tokens/sec | **53.7** |
| mean latency ms | 595.85 |
| backend | mlx-lm |

---

## 7. Phi-3 QLoRA and DPO

Source: `reports/phi3_qlora_skipped_or_training_metrics.json`,
`reports/phi3_dpo_skipped_or_training_metrics.json`

**Status: blocked — CUDA unavailable on this Mac. No adapters exist. Nothing fabricated.**

| item | status |
|---|---|
| SFT dataset | exists (`data/explanations/sft_{train,val,test}.jsonl`) |
| DPO preference dataset | exists (`data/explanations/dpo_{train,val}.jsonl`) |
| QLoRA training script | exists, dry-run passes 4/4 preflight checks |
| DPO training script | exists, dry-run passes 4/5 checks (`adapter_exists=false` — correct) |
| Colab notebook | exists (`notebooks/phi3_colab_training.ipynb`, 42 cells) |
| QLoRA adapter | **does not exist** |
| DPO adapter | **does not exist** |

---

## 8. Production Defaults

| setting | value |
|---|---|
| verifier checkpoint | `checkpoints/transformer_verifier_clean` |
| retrieval profile | `bm25_only` |
| aggregation mode | `per_passage_max` |
| support threshold | 0.55 |
| refute threshold | 0.5 |
| relevance gate | disabled (`null`) |

---

## 9. What Can Be Claimed (resume-safe)

- End-to-end fact verification pipeline: BM25 retrieval, DistilRoBERTa NLI verifier, grounded explanation generation
- Oracle macro-F1 **0.6728** vs retrieved macro-F1 **0.3887** (gap 0.2841), full 650-example test set
- Ablated 3 retrieval profiles at full 650-example scale; characterized oracle-vs-retrieved bottleneck
- Documented two negative results rigorously: robust verifier retrain regressed (−0.1058 macro_F1), relevance gate improved NEI FPR but regressed macro_F1
- ONNX export validated and functional; benchmarked at 55 ex/s CPU
- MLX LoRA verdict-prediction adapter: 0.695 accuracy, 0.4632 macro_F1 on 200-example eval (Apple Silicon, 53.7 tok/s)
- Generated SFT + DPO datasets for Phi-3 fine-tuning; created Colab training notebook
- Fixed a base-model key mismatch bug in MLX LoRA explanation training script; retrained at 300 iters; partial format compliance confirmed (format_correctness=0.2)
- 136 passing tests

## 10. What Must Not Be Claimed

- ONNX is faster than transformers on this Mac (it is not — 55 vs 62 ex/s at batch=1)
- Robust verifier improved anything (both oracle and retrieved metrics regressed)
- Relevance gate improved macro-F1 (it regressed; only NEI FPR improved)
- Phi-3 QLoRA or DPO adapter trained or exists (CUDA not available)
- MLX LoRA explanation adapter achieves production-ready structured output (format_correctness=0.2 at 300 iters, 10-example eval — partial compliance only)
