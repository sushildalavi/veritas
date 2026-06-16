# Veritas — Final Results

All numbers measured on this hardware unless otherwise noted. Sources are report files
in `reports/`. No results are fabricated or estimated.

---

## Verifier Performance (DistilRoBERTa NLI)

**Checkpoint:** `checkpoints/transformer_verifier_clean`
**Dataset:** 650-example test set (`fever_test_large` + `scifact_test_large`)
**Source:** `reports/oracle_vs_retrieved_v2_full.json`

| evidence mode | aggregation | accuracy | macro-F1 | NEI FPR |
|---|---|---|---|---|
| oracle | bundle | 0.7354 | 0.7246 | 0.0 |
| oracle | per_passage_max | 0.6985 | **0.6728** | 0.0 |
| retrieved | bundle | 0.3600 | 0.3332 | 0.8839 |
| retrieved | per_passage_max | 0.4062 | **0.3887** | 0.7098 |

Oracle→retrieved gap (per_passage_max): accuracy −0.2923, macro-F1 **−0.2841**

Per-class F1 on retrieved per_passage_max:
- SUPPORTED: 0.3211 — REFUTED: 0.4974 — NOT ENOUGH INFO: 0.3476

---

## Retrieval Profile Comparison

**Dataset:** 650-example test set
**Source:** `reports/retrieval_profile_comparison_650.json`

| profile | recall@10 | nDCG@10 | verifier macro-F1 |
|---|---|---|---|
| **bm25_only** (default) | **0.5334** | **0.4816** | **0.3887** |
| hybrid_bm25_dense | 0.5113 | 0.4288 | 0.3864 |
| hybrid_bm25_sentence_transformer | 0.5714 | 0.5234 | 0.3776 |

Key finding: better retrieval recall does not monotonically improve verifier
macro-F1. `hybrid_bm25_sentence_transformer` improves recall@10 by +0.038 but
reduces verifier macro-F1 by −0.0111 at more than 2× the runtime.

---

## Error Analysis (650-example test set)

**Source:** `reports/error_analysis_650.json`, `reports/error_analysis_650.md`

| failure bucket | count | pct |
|---|---|---|
| oracle_correct_retrieved_wrong | 222 | 34.15% |
| oracle_wrong_retrieved_wrong | 164 | 25.23% |
| oracle_wrong_retrieved_correct | 50 | 7.69% |
| oracle_correct_retrieved_correct | 214 | 32.92% |

The largest failure bucket (34%) is where retrieval failure flips an otherwise
correct verdict — confirming retrieval as the primary ceiling, not verifier capacity.

By dataset:
- FEVER retrieved macro-F1: 0.4392
- SciFact retrieved macro-F1: 0.1817

SciFact underperforms despite higher recall@10 (0.63 vs 0.51 for FEVER) — evidence
of a dataset-specific verifier weakness beyond the shared retrieval ceiling.

---

## Negative Results

### 1. Robust Verifier Retrain

**Source:** `reports/verifier_robustness_training_result.md`

| checkpoint | evidence | macro-F1 | delta |
|---|---|---|---|
| baseline | oracle | 0.6728 | — |
| baseline | retrieved | 0.3887 | — |
| robust retrain | oracle | 0.4376 | −0.2352 |
| robust retrain | retrieved | 0.2829 | −0.1058 |

Both metrics regressed. Root cause: combined training set was 55% NEI (76% NEI in
retrieved pairs) without class reweighting — model collapsed toward NEI.
**Production checkpoint unchanged.**

### 2. Relevance Gate (lexical token overlap)

**Source:** `reports/oracle_vs_retrieved_v2_full_gated.json`

| mode | macro-F1 | NEI FPR |
|---|---|---|
| no gate | 0.3887 | 0.7098 |
| gate=0.5 | 0.3557 (−0.0330) | 0.2857 (−0.4241) |

NEI FPR improved substantially. End-to-end macro-F1 regressed.
**Gate disabled by default** (`relevance_gate_threshold: null`).

---

## Inference Benchmark

**Source:** `reports/verifier_inference_benchmark.json`,
`reports/onnx_verifier_benchmark.json`

### DistilRoBERTa verifier (transformers native)

| batch | device | latency ms | throughput ex/s |
|---|---|---|---|
| 1 | CPU | 16.0 | 62.4 |
| 8 | CPU | 58.9 | 135.8 |
| 1 | MPS | 6.5 | **153.8** |
| 8 | MPS | 23.5 | **340.9** |

### ONNX export (CPU only)

| batch | latency ms | throughput ex/s |
|---|---|---|
| 1 | 18.2 | 55.1 |
| 8 | 133.9 | 59.7 |

**ONNX is slower than native transformers on this Mac** (55 vs 62 ex/s at batch=1).
ONNX export is valid and functional; the latency benefit applies on CUDA hardware.

---

## MLX LoRA (Apple Silicon)

### Verdict-prediction adapter (`checkpoints/mlx_lora_verifier`)

**Source:** `reports/mlx_lora_eval_200.json`
Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, 100 iters, 53.7 tok/s

| metric | value |
|---|---|
| verdict_accuracy | **0.695** |
| macro-F1 | **0.4632** |
| citation_valid_rate | 0.6 |
| parseable_rate | 1.0 |
| SUPPORTED F1 | 0.7024 |
| REFUTED F1 | 0.6872 |
| NOT ENOUGH INFO F1 | 0.0 |

NEI F1=0.0: the model rarely predicts NEI on oracle-evidence training data.

### Explanation adapter (`adapters/mlx_qwen_veritas_lora`)

**Source:** `reports/mlx_lora_500_eval.json`, `reports/mlx_lora_generation_fix.md`
Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, 500 iters, 256 examples

A base-model key mismatch bug (`base_model` config key vs mlx_lm's `model` key) caused
training to silently use `Qwen/Qwen3-0.6b` — generating a dimension-mismatched adapter.
Fixed. Retrained at 300 iters, then extended to 500 iters.

| metric | base model | adapter (300 iters) | adapter (500 iters) |
|---|---|---|---|
| format_correctness | 0.0 | 0.2 | **0.28** |
| citation_presence | 0.0 | 0.1 | **0.72** |
| decision_label_consistency | 0.0 | 0.1 | **0.24** |
| avg explanation length (words) | 0.0 | 34.2 | 33.2 |

*25-example eval. Significant improvement in citation_presence (0.1→0.72). Not
production-grade — partial format compliance. Prompt leakage suppressed by
stop-token trimming in eval script.*

---

## Phi-3 QLoRA and DPO

**Status: infrastructure complete, training blocked on CUDA.**

| artifact | status |
|---|---|
| SFT dataset | exists (`data/explanations/sft_{train,val,test}.jsonl`) |
| DPO preference dataset | exists (`data/explanations/dpo_{train,val}.jsonl`) |
| QLoRA training script | exists; dry-run passes all 4 preflight checks |
| DPO training script | exists; dry-run passes 4/5 checks (adapter_exists=false is correct) |
| Colab notebook | exists (`notebooks/phi3_colab_training.ipynb`, 42 cells) |
| QLoRA adapter | **does not exist** (CUDA unavailable on this Mac) |
| DPO adapter | **does not exist** (requires QLoRA first) |

---

## Production Defaults

| setting | value |
|---|---|
| verifier checkpoint | `checkpoints/transformer_verifier_clean` |
| retrieval profile | `bm25_only` |
| aggregation mode | `per_passage_max` |
| support threshold | 0.55 |
| refute threshold | 0.5 |
| relevance gate | disabled (`null`) |

---

## Resume-Safe Summary

> Built an end-to-end open-domain fact-verification pipeline using BM25 retrieval and
> DistilRoBERTa NLI; oracle macro-F1 0.67 vs retrieved 0.39 on 650 FEVER+SciFact
> examples, with a structured ablation of three retrieval strategies, two negative
> results (verifier retrain, relevance gate), ONNX export benchmarked at 55 ex/s,
> and an MLX LoRA adapter achieving 0.695 verdict accuracy on Apple Silicon at
> 53.7 tok/s.

**What can be claimed:**
- End-to-end pipeline with BM25 + DistilRoBERTa NLI + grounded explanation generation
- Oracle macro-F1 0.6728, retrieved macro-F1 0.3887, gap 0.2841, 650-example test set
- 3 retrieval profiles ablated at full scale
- 2 negative results rigorously documented (robust retrain −0.1058 F1, gate −0.0330 F1)
- ONNX export validated and benchmarked
- MLX LoRA verdict-prediction: 0.695 accuracy, 53.7 tok/s, Apple Silicon
- SFT + DPO datasets and Colab notebook for Phi-3 fine-tuning
- Fixed base-model key mismatch bug in MLX LoRA training script

**What must not be claimed:**
- ONNX faster on this Mac (it is not)
- Robust retrain or relevance gate improved macro-F1 (both regressed)
- Phi-3 QLoRA/DPO adapter exists (it does not)
- MLX explanation adapter achieves production-quality output (partial compliance, 0.2)
