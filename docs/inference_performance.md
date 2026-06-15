# Inference Performance Lab

This document describes Veritas's inference benchmarking suite: reproducible,
honest latency/throughput measurements for the verifier and explanation-model
serving paths, plus scaffolding for GPU-only runtimes (ONNX, Triton) that are
not available in the current development environment.

All numbers below were measured on a local Apple Silicon machine (CPU + MPS,
no CUDA). Nothing here is fabricated or extrapolated -- benchmarks that
require unavailable hardware/services are written as `status: skipped` with
the exact command needed to run them on a suitable host.

## Verifier inference benchmark

`scripts/benchmark_verifier_runtime.py` loads the local
`checkpoints/transformer_verifier_clean` checkpoint (a fine-tuned
RoBERTa sequence classifier, the same one used in production serving) with
`transformers`/`torch` and measures forward-pass latency and throughput
across batch sizes on every available device.

```bash
python3 scripts/benchmark_verifier_runtime.py
```

Config: `configs/inference_benchmark.yaml` (`verifier:` section) --
batch sizes, devices, iteration counts, and the sample claim/passage used to
build the input text are all configurable.

Output: `reports/verifier_inference_benchmark.json` / `.md`.

Latest measured results (CPU + MPS, batch sizes 1/8/32):

| device | batch_size | mean_latency_ms | throughput (examples/sec) |
| --- | --- | --- | --- |
| cpu | 1 | 16.0 | 62.4 |
| cpu | 8 | 58.9 | 135.8 |
| cpu | 32 | 208.3 | 153.6 |
| mps | 1 | 6.5 | 153.8 |
| mps | 8 | 23.5 | 340.9 |
| mps | 32 | 87.8 | 364.4 |

Takeaways: MPS gives a ~2.4x throughput improvement over CPU at batch size 32,
and batching from 1 to 32 improves throughput by ~2.4x on both devices --
i.e. the model is still latency-bound rather than memory-bandwidth-bound at
these batch sizes on this hardware.

## Explanation-serving benchmark

`scripts/benchmark_inference_serving.py` benchmarks two explanation backends:

1. **`transformers` (local generation)**: loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
   (already cached locally) and measures single-request generation latency
   and tokens/sec for a fixed prompt and `max_new_tokens`, on every available
   device. This is always measured -- no GPU or external service required.
2. **`vllm`**: health-checks the OpenAI-compatible vLLM endpoint configured in
   `configs/vllm_serving.yaml`. If reachable, measures request throughput at
   several concurrency levels via `serving.vllm_client.VllmExplanationGenerator`.
   If unreachable (the default in this environment), the section is written
   as `status: skipped` with the reason and next steps.

```bash
python3 scripts/benchmark_inference_serving.py
```

Output: `reports/inference_benchmark.json` / `.md`.

Latest measured results:

| device | mean_latency_ms | mean_generated_tokens | tokens/sec |
| --- | --- | --- | --- |
| cpu | 1959.5 | 32 | 16.3 |
| mps | 1040.9 | 32 | 30.7 |

vLLM: `status: skipped` (`vllm endpoint unavailable in current environment`).

### Running the vLLM benchmark on a GPU host

1. Start a vLLM OpenAI-compatible server (see `docs/vllm_serving.md` /
   `scripts/serve_vllm_explanations.py`), e.g.:

   ```bash
   python3 -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-1.5B-Instruct --port 8001
   ```

2. Point `configs/vllm_serving.yaml` at the running server (`vllm.base_url`,
   `vllm.model`), then re-run:

   ```bash
   python3 scripts/benchmark_inference_serving.py
   ```

   The `vllm` section of the report will switch from `skipped` to `measured`
   and include per-concurrency throughput.

## ONNX verifier runtime

See `docs/inference_runtime_landscape.md` for the ONNX/Triton scaffolding
status and how to run those benchmarks on a host with `onnxruntime` /
`triton` / CUDA installed.

## What is measured vs scaffolded

| Component | Status | Notes |
| --- | --- | --- |
| Verifier (PyTorch, CPU/MPS) | Measured | `reports/verifier_inference_benchmark.json` |
| Explanation generation (transformers, CPU/MPS) | Measured | `reports/inference_benchmark.json` |
| Explanation generation (vLLM) | Skipped (no server running) | Runnable once a vLLM server is started |
| Verifier (ONNX Runtime) | Scaffolded, not run | `onnxruntime` not installed locally |
| Dense-scoring microbenchmark (Triton/CUDA) | Scaffolded, not run | No CUDA/Triton locally |
| SGLang / MLC / FlashAttention / TVM / MLIR | Architecture notes only | See `docs/inference_runtime_landscape.md` |
