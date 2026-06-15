# Inference Runtime Landscape (Architecture Notes)

This document is **architecture-awareness only** -- no implementation, no
benchmarks, no claimed numbers. It maps several inference-runtime
technologies to Veritas's actual bottlenecks (the verifier forward pass and
explanation-model generation, both measured in
`docs/inference_performance.md`) and notes what adopting each would require.
It complements the ONNX/Triton scaffolds (`serving/onnx_verifier_backend.py`,
`scripts/benchmark_triton_dense_scoring.py`), which are runnable but skip on
this CPU/MPS-only machine.

## SGLang

SGLang is a serving framework for LLMs built around **structured generation**
(constrained decoding to a grammar/JSON schema) and **RadixAttention**, a
prefix-caching scheme that reuses KV-cache across requests sharing a common
prompt prefix.

Relevance to Veritas: the explanation generator already requires strict JSON
output (`strict_json_output: true` in `configs/serving.yaml`,
`rag/explanation_generator.py`'s `_parse_json_response`). Today this is
enforced post-hoc by parsing the model's text output and falling back to a
template if parsing fails. SGLang's constrained decoding would guarantee
valid `{"explanation": ..., "citations": [...]}` output at generation time,
removing the parse-failure fallback path entirely. RadixAttention would also
help because every explanation prompt shares a long common prefix (the
`EXPLANATION_PROMPT` template + retrieved evidence block) across requests for
different claims.

Adoption would require: running SGLang's server in place of (or alongside)
vLLM, defining a JSON schema for the explanation output, and pointing
`serving/vllm_client.py`-equivalent client code at SGLang's OpenAI-compatible
or native endpoint. No verifier changes needed.

## MLC-LLM / TVM Unity

MLC-LLM compiles LLMs (via TVM's Unity flow) to run efficiently across
backends -- CUDA, Metal, WebGPU, mobile NPUs -- from a single model
definition.

Relevance to Veritas: the verifier checkpoint
(`checkpoints/transformer_verifier_clean`, a RoBERTa sequence classifier) and
the small explanation model (TinyLlama-1.1B, benchmarked in
`reports/inference_benchmark.json`) are both small enough to be attractive
edge/on-device deployment targets. MLC-LLM's Metal backend is directly
relevant here since this development machine is Apple Silicon (MPS is
currently used via plain PyTorch, which is already ~2.4x faster than CPU at
batch size 32 per `reports/verifier_inference_benchmark.md` -- a
TVM/MLC-compiled Metal kernel could plausibly improve on that further, but
this has not been measured).

Adoption would require: exporting the verifier to TVM's IR (via the ONNX
export path in `scripts/export_verifier_onnx.py` as an intermediate step,
then `relax.frontend.onnx` import), defining a compilation target, and
building a thin Python binding to call the compiled module from
`serving/onnx_verifier_backend.py`'s call sites.

## FlashAttention / FlashInfer

FlashAttention is a fused, IO-aware attention kernel that avoids materializing
the full attention matrix in GPU memory; FlashInfer extends this with
batching/paging optimizations tailored to LLM serving (e.g. PagedAttention-style
KV-cache layouts).

Relevance to Veritas: both the verifier (RoBERTa, max_length=512) and the
explanation model (TinyLlama, 32-token generations) use standard attention via
`transformers`. At the current sequence lengths and batch sizes (measured up
to batch=32 for the verifier, batch=1 for generation), attention is unlikely
to be the dominant cost compared to the feed-forward layers -- FlashAttention's
benefit grows with sequence length and batch size. It becomes directly
relevant if `max_evidence`/`final_top_k` were increased (longer verifier
inputs) or if explanation generation moved to longer outputs or larger batch
sizes.

Adoption would require: CUDA (FlashAttention's optimized kernels are
CUDA-only; the MPS/CPU path used here does not benefit) and either
`flash-attn`-compatible model code or a `transformers` build with
`attn_implementation="flash_attention_2"`.

## TVM / MLIR (general compiler stack)

TVM and MLIR represent the general "compile the model graph to optimized
device code" approach, of which TVM Unity (above) is one instance. MLIR
specifically is the IR layer underneath several of these stacks (including
parts of TVM, ONNX Runtime's execution providers, and PyTorch's `torch.compile`
via `torch-mlir`/Inductor).

Relevance to Veritas: `torch.compile` (PyTorch's own MLIR/Inductor-backed
compiler) is the lowest-friction entry point -- it requires no new export
format and can be applied directly to the loaded verifier model in
`scripts/benchmark_verifier_runtime.py` (`model = torch.compile(model)`)
before benchmarking. This was not done here to keep the baseline numbers
simple and directly comparable across batch sizes, but it is the natural next
benchmark variant to add once the current numbers are reviewed.

## Summary

| Technology | Status in Veritas | Why it's relevant |
| --- | --- | --- |
| ONNX Runtime | Scaffolded (`serving/onnx_verifier_backend.py`), skips without `onnxruntime` | Portable CPU inference for the verifier |
| Triton (GPU kernels) | Scaffolded (`scripts/benchmark_triton_dense_scoring.py`), skips without CUDA | Dense-retrieval scoring kernel example |
| vLLM | Implemented (`serving/vllm_client.py`), skips without a running server | Explanation generation serving |
| SGLang | Architecture note only | Constrained JSON decoding + prefix caching for explanations |
| MLC-LLM / TVM Unity | Architecture note only | Edge/Metal deployment of verifier + small LM |
| FlashAttention / FlashInfer | Architecture note only | Needs CUDA + longer sequences/batches than currently used |
| TVM / MLIR (`torch.compile`) | Architecture note only | Lowest-friction compiler benchmark to add next |
