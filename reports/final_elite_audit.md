# Veritas — Final Elite Audit

Veritas | Neural Fact Verification with Hybrid Retrieval, Cross-Encoder
Ranking, Fine-Tuned DistilRoBERTa & MLX LoRA

GitHub: https://github.com/sushildalavi/veritas
Live demo: https://sushildalavi-veritas.hf.space

**PROJECT NOT FULLY COMPLETE: CUDA QLoRA and DPO are ready but not run (no
CUDA GPU on this machine).**

---

## 1. Executive Summary

Veritas is an end-to-end factual claim verification pipeline: BM25 + dense
retrieval, hybrid RRF fusion, cross-encoder reranking, a class-weighted
DistilRoBERTa verifier, template-based citation-grounded explanation
generation, citation faithfulness checking, and FastAPI/Gradio serving with
config-driven backend routing. All components through serving and
evaluation are real, reproducible, and backed by checked-in reports. An MLX
LoRA adapter was also trained on Apple Silicon and evaluated on 200 held-out
examples (verdict accuracy 0.695, macro F1 0.4632), with a 300-iteration
adapter trained for comparison and rejected as overfit. A 1,382-pair DPO
preference dataset has been built from the verifier training data. The two
remaining phases — CUDA QLoRA fine-tuning and DPO alignment — require a CUDA
GPU with `bitsandbytes`, which this development machine does not have.
Ready-to-run scripts, configs, and Kaggle/Colab notebooks are checked in for
both, but no adapters or eval reports exist, so those phases are honestly
marked incomplete.

## 2. Project Goals

- Build a reproducible fact-verification pipeline on FEVER/SciFact samples.
- Evaluate retrieval (BM25 vs. dense vs. hybrid) at meaningful scale (200
  queries, 9,804-passage corpus).
- Evaluate cross-encoder reranking against BM25/dense/learned rankers.
- Train a class-weighted transformer verifier that improves materially over
  a sklearn TF-IDF/LogReg baseline, including on the minority REFUTED class.
- Measure the oracle-vs-retrieved generalization gap honestly.
- Generate citation-grounded explanations and measure faithfulness.
- Fine-tune an LLM explanation generator with LoRA on Apple Silicon (MLX)
  and build a DPO preference dataset for future alignment.
- (Stretch, GPU-gated) Fine-tune an LLM explanation generator with CUDA
  QLoRA and align it with DPO.
- Serve everything through FastAPI + Gradio with health/metrics endpoints
  exposing the active model and backend configuration.

## 3. Architecture Overview

```mermaid
flowchart LR
    A[Claim] --> B[BM25 + Dense Retrieval]
    B --> C[Hybrid RRF]
    C --> D[Cross-Encoder Reranker]
    D --> E[DistilRoBERTa Verifier]
    E --> F[Template Explanation]
    F --> G[Citation Checker]
    G --> H[FastAPI / Gradio Serving]
```

Key modules: `retrieval/` (BM25, dense, hybrid), `ranking/` (cross-encoder,
learned, heuristic), `models/` (verifier router, sklearn + transformer
verifiers, QLoRA/DPO stubs), `rag/` (context builder, template explanation,
citation checker), `serving/` (FastAPI app, model loader, config routing),
`ui/` (Gradio demo).

## 4. Data Pipeline & Quality

- Source: sampled FEVER + SciFact, processed into `data/processed/*_large.jsonl`.
- Evidence corpus: 9,804 passages (`evidence_corpus_large.jsonl`).
- Clean verifier dataset built from gold evidence: 2,809 train / 650 val / 650
  test examples (`data/processed/verifier_{train,val,test}.jsonl`), audited
  in `reports/verifier_data_audit.md`.
- Data quality summary: `reports/data_quality_large.{json,md}` (4,109 sampled
  records).

## 5. Retrieval Evaluation (Phase 3)

- 200 validation queries against the 9,804-passage corpus
  (`reports/retrieval_eval_neural_large.md`).
- BM25: MRR 0.398, nDCG@10 0.389, recall@10 0.442.
- Dense (`sentence-transformers/all-MiniLM-L6-v2`): MRR 0.534, nDCG@10 0.506,
  recall@10 0.524.
- Hybrid (RRF): MRR 0.491, nDCG@10 0.478, recall@10 0.535 — best recall@10 of
  the three.

## 6. Ranking Evaluation (Phase 4)

- 200 validation queries, candidate_k=10
  (`reports/ranking_eval_cross_encoder_large.md`).
- Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`): MAP 0.540, MRR
  0.562, nDCG@10 0.565.
- Cross-encoder + learned features: MAP 0.584, MRR 0.587, nDCG@10 0.604 —
  best of all rankers evaluated (BM25, dense, learned, heuristic, RRF).

## 7. Verifier Training & Evaluation (Phases 1-2)

- Sklearn TF-IDF/LogReg baseline (`checkpoints/verifier_clean/`): test
  accuracy 0.486, macro F1 0.484 (`reports/verifier_clean_baseline.md`).
- Class-weighted DistilRoBERTa (`checkpoints/transformer_verifier_clean/`):
  test accuracy 0.718, macro F1 0.711, REFUTED recall 0.745
  (`reports/transformer_verifier_clean_eval.md`).
- Oracle evaluation (gold evidence): accuracy 0.717, macro F1 0.710
  (`reports/oracle_verifier_eval.md`).
- End-to-end evaluation (BM25 top-1 retrieved evidence): accuracy 0.440,
  macro F1 0.414 — a 0.277 accuracy / 0.296 macro-F1 gap vs. oracle
  (`reports/end_to_end_verifier_eval.md`). This gap is real and is reported
  rather than hidden.

## 8. Serving & Backend Routing (Phase 5)

- `serving/model_loader.py` resolves the verifier checkpoint with priority:
  DeBERTa clean > DistilRoBERTa clean > legacy override > old transformer
  smoke checkpoint > sklearn > mock. Default resolves to
  `checkpoints/transformer_verifier_clean` (macro F1 0.711, `fallback_used=False`).
- `/health`, `/verify`, and `/metrics` expose `verifier_backend`,
  `model_name`, `checkpoint_path`, `verifier_macro_f1`, `retrieval_backend`,
  `embedding_model`, `reranker_backend`, `cross_encoder_model`, and fallback
  flags for both retrieval and reranking.
- Gradio UI (`ui/app.py`) surfaces `model_name` alongside verdict,
  confidence, backend, and citation status.

## 8a. MLX LoRA Alignment (Apple Silicon, local) — COMPLETE

**Status: COMPLETE.** A real LoRA adapter was trained with `mlx_lm.lora
--fine-tune-type lora` on `mlx-community/Qwen2.5-1.5B-Instruct-4bit` using
600 chat-formatted training examples (`data/processed/mlx_lora/train.jsonl`).

- `checkpoints/mlx_lora_verifier/` — 100-iteration adapter (best). Evaluated
  on 200 held-out examples: verdict accuracy 0.695, macro F1 0.4632,
  per-class F1 (SUPPORTED 0.7024, REFUTED 0.6872, NOT ENOUGH INFO 0.0),
  citation valid rate 0.6, unsupported-sentence rate 0.2537, mean latency
  1.2984s/example (`reports/mlx_lora_eval_200.{json,md}`).
- `checkpoints/mlx_lora_verifier_300/` — 300-iteration adapter, trained and
  evaluated for comparison on the same 200 examples: verdict accuracy
  dropped to 0.52, macro F1 to 0.24 (REFUTED F1 collapsed to 0.04) —
  overfitting, so the 100-iteration adapter remains the best
  (`reports/mlx_lora_comparison.{json,md}`).
- A 600-iteration run was not attempted: per the stop condition, once the
  300-iteration adapter showed clear overfitting on this 600-example
  training set, training longer would only extend the same trend.

## 9. CUDA QLoRA Fine-Tuning (Phase 6) — READY, NOT RUN

**Status: NOT COMPLETE (READY_NOT_RUN).** `torch.cuda.is_available()` is
`False` and `bitsandbytes` is not installed/supported on this machine (Apple
Silicon, MPS only). Per project rules, no adapter files or `cuda_qlora_eval`
reports were fabricated. Delivered instead:

- `configs/cuda_qlora_tinyllama.yaml` — full training config for
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (4-bit NF4 quantization, LoRA r=16/
  alpha=32/dropout=0.05 on q/k/v/o projections).
- `scripts/train_cuda_qlora.py` — complete, ready-to-run training script
  (guards on CUDA/bitsandbytes, builds the SFT dataset from
  `data/processed/mlx_lora/{train,valid}.jsonl` via
  `evaluation.cuda_verifier_eval`, trains via `transformers.Trainer`, saves
  the adapter to `checkpoints/cuda_qlora_verifier/`, evaluates verdict
  accuracy/macro F1/citation validity via the shared
  `evaluation.cuda_verifier_eval.evaluate_adapter`, writes
  `reports/cuda_qlora_eval.{json,md}`). `--export-sft-only` was run locally
  (no GPU) and produced real `data/processed/sft_{train,val}.jsonl` files
  (600/100 examples).
- `notebooks/12_cuda_qlora_kaggle_colab.ipynb` — installs dependencies,
  checks for a CUDA GPU, clones the repo, runs training + evaluation, zips
  and downloads the adapter + reports.
- `reports/cuda_qlora_READY.md` — full readiness documentation and run
  instructions for Colab/Kaggle.

## 10. CUDA DPO Alignment (Phase 7) — READY, NOT RUN

**Status: NOT COMPLETE (READY_NOT_RUN).** DPO depends on the CUDA QLoRA
adapter from Phase 6, which does not exist on this machine, and on
`bitsandbytes`/`trl`, which are not installed. Per project rules, no adapter
files or `cuda_dpo_eval` reports were fabricated. Delivered instead:

- `data/processed/preference_pairs.jsonl` — 1,382 `{"prompt", "chosen",
  "rejected"}` pairs built from `data/processed/verifier_train.jsonl` by
  `scripts/build_preference_pairs_real.py`. `chosen` is a correct verdict
  with a citation-valid grounded explanation (chosen-citation-valid rate
  1.0); `rejected` is one of wrong verdict, invalid citation, or unsupported
  explanation (461/461/460 split). See
  `reports/preference_pair_stats.{json,md}`. **This is dataset construction
  only — DPO training has not been run.**
- `configs/cuda_dpo_tinyllama.yaml` — DPO hyperparameters (`beta=0.1`,
  `max_length=512`, `max_prompt_length=384`) and paths.
- `scripts/train_cuda_dpo.py` — complete, ready-to-run script that verifies
  the CUDA QLoRA adapter + CUDA/`bitsandbytes`/`peft`/`trl` are available,
  loads the QLoRA adapter as the starting policy, evaluates it on
  `data/processed/mlx_lora/valid.jsonl` **before** DPO, runs
  `trl.DPOTrainer` on `preference_pairs.jsonl` (`ref_model=None`), saves
  `checkpoints/cuda_dpo_verifier/`, evaluates **after** DPO, and writes
  `reports/cuda_dpo_eval.{json,md}` with a before/after/delta table
  (verdict accuracy, macro F1, citation valid rate, unsupported-sentence
  rate, verdict consistency rate).
- `notebooks/13_cuda_dpo_kaggle_colab.ipynb` — installs dependencies, checks
  for a CUDA GPU, clones the repo, loads the uploaded CUDA QLoRA adapter,
  runs DPO training + before/after evaluation, zips and downloads the
  adapter + reports.
- `reports/cuda_dpo_READY.md` — full readiness documentation and run
  instructions for Colab/Kaggle, superseding the older
  `reports/dpo_BLOCKED_QLORA_REQUIRED.md`.

## 11. RAG / Explanation Generation

- `rag.build_context` assembles citation-numbered evidence blocks.
- `rag.generate_template_explanation` produces a deterministic, citation-
  grounded explanation from the top evidence item and predicted verdict.
- `rag.check_citations` validates that cited evidence IDs exist, computes
  citation precision, unsupported-sentence rate, and verdict consistency.

## 12. Citation Faithfulness (Phase 8)

- `reports/faithfulness_comparison.{json,md}` measures the template
  explanation generator (driven by the clean DistilRoBERTa verifier) on 200
  real validation examples: citation valid rate 0.560, mean citation
  precision 1.000, mean unsupported-sentence rate 0.306, verdict consistency
  rate 0.755.
- CUDA QLoRA and CUDA DPO explanation generators are recorded as
  `"not trained - GPU required"` / `"not trained - QLoRA required"` rather
  than fabricated.

## 13. Pareto / Tradeoff Analysis (Phase 9)

- `reports/final_pareto_analysis.{json,md}` compares the two trained
  verifiers on real test-set macro F1, measured latency, and checkpoint size:
  - `distilroberta-clean`: macro F1 0.711, ~29.7ms/example, ~316.7MB.
  - `sklearn-tfidf-logreg`: macro F1 0.484, ~0.017ms/example, ~0.6MB.
  - Both are on the Pareto frontier — DistilRoBERTa for quality, sklearn for
    deployment cost.
- `cuda-qlora-tinyllama` and `cuda-dpo-tinyllama` are listed as
  `"not trained - GPU required"` / `"not trained - QLoRA required"`.

## 14. Deployment (Hugging Face Space)

- `DEPLOYMENT.md` documents lightweight (default, `configs/serving.yaml`:
  BM25-only retrieval, no reranking, auto-resolved verifier) and advanced
  (`configs/serving_advanced.yaml`: hybrid dense retrieval + cross-encoder
  reranking) serving modes.
- Live demo: https://sushildalavi-veritas.hf.space.

## 15. Testing & CI

- `make test`: 73 passed (pytest), run after every phase.
- `make lint`: `compileall` across all source packages, run after every
  phase.
- CI kept lightweight (no GPU-dependent jobs).

## 16. Reproducibility

- All training/eval scripts accept explicit seeds (seed=42), record git
  commit hashes, Python/library versions, and sample sizes in
  `metadata`/checkpoint `metadata.json`.
- Evaluation scripts (`scripts/evaluate_oracle_vs_retrieved.py`,
  `scripts/eval_faithfulness_final.py`, `scripts/pareto_analysis_final.py`,
  `scripts/run_retrieval_eval.py`, `scripts/run_ranking_eval.py`) are
  deterministic given fixed inputs and run on CPU.

## 17. Known Limitations

- Sample-scale data (2,809/650/650 verifier examples; 200-query retrieval/
  ranking evals), not full FEVER/SciFact benchmarks.
- Large oracle-vs-retrieved generalization gap (macro F1 0.710 → 0.414) —
  the verifier is much weaker when fed real retrieved evidence than gold
  evidence.
- Template explanations are citation-valid only 56% of the time on the
  measured sample.
- The MLX LoRA adapter (100 iterations, 200-example eval) has macro F1
  0.4632 with a NOT ENOUGH INFO F1 of 0.0; a 300-iteration adapter overfit
  and was rejected (`reports/mlx_lora_comparison.md`).
- A DPO preference-pair dataset (1,382 pairs) has been built, but DPO
  training has not been run on either path.
- CUDA QLoRA and CUDA DPO are not trained; no adapters or eval reports
  exist for either, though configs/scripts/notebooks are ready.

## 18. Final Completion Status

**PROJECT NOT FULLY COMPLETE: CUDA QLoRA/DPO are ready but not run (require
GPU execution).**

Phases 1-5, 6c-6d, and 8-11 (data, retrieval, ranking, verification, serving,
MLX LoRA alignment with 200-example evaluation, DPO preference-pair dataset,
faithfulness, Pareto analysis, deployment docs, this audit) are complete with
real, checked-in artifacts. Phases 6-7 (CUDA QLoRA, CUDA DPO) and 7b (MLX
DPO) are blocked on GPU/trainer availability and are documented honestly as
incomplete, with ready-to-run scripts/configs/notebooks checked in for when
GPU access is available. See `reports/final_completion_gate.md` for the
itemized yes/no checklist.
