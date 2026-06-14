# LLM Alignment Experiments Archive

This directory preserves an earlier experimental path for historical reference only. The files
here are not part of the final Veritas architecture and are not exercised by `make test` or
`make lint`.

## What this was

An early direction explored aligning the explanation/verdict model with DPO and fine-tuning a
local LLM with QLoRA (`microsoft/phi-2`, TinyLlama). Config-parsing scaffolding and model wrappers
were built for this path:

- `configs/dpo.yaml`, `configs/qlora_phi.yaml`, `configs/qlora_tinyllama.yaml`
- `training/train_dpo.py`, `training/train_qlora.py`
- `models/dpo_model.py`, `models/qlora_llm.py`
- `scripts/train_qlora_real.py`
- `evaluation/cuda_verifier_eval.py`
- `tests/test_alignment.py`

A CUDA-dependent variant of this same direction was already archived separately in
[`docs/archive/cuda_experiments/`](/Users/sushildalavi/Desktop/Github/Veritas/docs/archive/cuda_experiments).

## Why it's not the final direction

QLoRA/DPO did not become the final project direction. The measured bottleneck in Veritas is
**evidence retrieval and ranking** (oracle verifier macro-F1 0.710 vs. retrieved-evidence
macro-F1 0.414) — not LLM alignment. The final architecture focuses on:

- trainable evidence retrieval (bi-encoder)
- cross-encoder reranking
- verifier robustness on retrieved evidence

LLM-based explanation generation (Qwen2.5 MLX LoRA + preference-guided reranking) remains in the
main codebase as an **optional** explanation layer. It is not the source of truth for the verdict
— the trained verifier is.

## Note on imports

These files used absolute package imports (e.g. `from models.dpo_model import ...`) that assumed
they lived inside the installed package. They will not run from this archive location without
adjusting import paths; they are kept as reference only.
