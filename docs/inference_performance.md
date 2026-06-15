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

## Mac-local LLM inference backends

`scripts/benchmark_mac_local_inference.py` benchmarks LLM generation backends
that run natively on Apple Silicon without CUDA, as alternatives to the
vLLM/Triton paths above (which remain CUDA-only and skip on this machine):

- **mlx-lm**: Apple's MLX framework (Metal/GPU). Measured if the `mlx-lm`
  package is installed.
- **Ollama**: measured if an Ollama server is reachable at
  `mac_local_backends.ollama.base_url` (default `http://localhost:11434`).
- **llama.cpp**: measured if `mac_local_backends.llama_cpp.binary_path` and
  `.model_path` are configured and point to an existing Metal-enabled binary
  and GGUF model.

```bash
python3 scripts/benchmark_mac_local_inference.py
```

Output: `reports/mac_local_inference_benchmark.json` / `.md`.

Latest measured results:

| backend | model | device | max_tokens | mean_latency_ms | p50_latency_ms | p95_latency_ms | tokens/sec | requests/sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx-lm | mlx-community/Qwen2.5-1.5B-Instruct-4bit | mps | 32 | 595.85 | 582.66 | 621.71 | 53.7 | 1.678 |

Ollama: `status: skipped` (`ollama endpoint http://localhost:11434 is unreachable in this environment`).
llama.cpp: `status: skipped` (`mac_local_backends.llama_cpp.binary_path / .model_path are not configured`).

Compared to the `transformers` CPU/MPS numbers above (TinyLlama-1.1B, 16.3
tok/s CPU / 30.7 tok/s MPS for 32 tokens), mlx-lm on a larger model
(Qwen2.5-1.5B, 4-bit) reaches 53.7 tok/s -- these are different model sizes
and quantizations, so this is not a same-model speedup comparison, just two
independently measured local backends.

### Enabling Ollama / llama.cpp

- **Ollama**: `ollama serve` (or launch the Ollama app), then
  `ollama pull qwen2.5:1.5b` (or update `mac_local_backends.ollama.model` in
  `configs/inference_benchmark.yaml` to a model you have pulled), then re-run
  the benchmark.
- **llama.cpp**: build with Metal support (`LLAMA_METAL=1`), download a GGUF
  model, then set `mac_local_backends.llama_cpp.binary_path` and `.model_path`
  in `configs/inference_benchmark.yaml`.

## ONNX verifier runtime

`scripts/export_verifier_onnx.py` exports the
`checkpoints/transformer_verifier_clean` checkpoint to ONNX, and
`scripts/benchmark_verifier_onnx.py` benchmarks it via
`onnxruntime.InferenceSession` (CPU execution provider), matching the batch
sizes used in the PyTorch verifier benchmark above.

```bash
pip install onnx onnxruntime onnxscript
python3 scripts/export_verifier_onnx.py
python3 scripts/benchmark_verifier_onnx.py
```

Output: `reports/onnx_export.json` / `.md`, `reports/onnx_verifier_benchmark.json` / `.md`.

Export note: with `torch` 2.11, `torch.onnx.export(..., dynamo=True)` (the
default) requires a dummy batch size > 1 and `dynamic_shapes` (rather than
`dynamic_axes`, which is ignored under `dynamo=True`) for the batch dimension
to remain dynamic in the exported graph -- otherwise the exported model only
accepts batch size 1. `scripts/export_verifier_onnx.py` now exports with a
dummy batch of 2 and `dynamic_shapes`. The exported model's outputs match the
PyTorch model to within 5.5e-6 absolute difference on a 3-example batch.

Latest measured results (CPU, batch sizes 1/8), compared against the PyTorch
CPU numbers from the table above:

| backend | batch_size | mean_latency_ms | throughput (examples/sec) |
| --- | --- | --- | --- |
| pytorch (cpu) | 1 | 16.0 | 62.4 |
| onnxruntime (cpu) | 1 | 18.2 | 55.1 |
| pytorch (cpu) | 8 | 58.9 | 135.8 |
| onnxruntime (cpu) | 8 | 133.9 | 59.7 |

On this machine, ONNX Runtime (default CPU execution provider, no further
tuning) is slower than the PyTorch CPU baseline at both batch sizes -- **no
ONNX speedup is claimed**. This may reflect PyTorch's Accelerate/MPS-tuned
kernels on Apple Silicon outperforming the default ONNX Runtime CPU provider;
it has not been investigated further (e.g. `onnxruntime` execution-provider
tuning, `CoreMLExecutionProvider`).

See `docs/inference_runtime_landscape.md` for the Triton scaffolding status
and how to run it on a host with CUDA installed.

## What is measured vs scaffolded

| Component | Status | Notes |
| --- | --- | --- |
| Verifier (PyTorch, CPU/MPS) | Measured | `reports/verifier_inference_benchmark.json` |
| Explanation generation (transformers, CPU/MPS) | Measured | `reports/inference_benchmark.json` |
| Explanation generation (vLLM) | Skipped (no server running) | Runnable once a vLLM server is started |
| Verifier (ONNX Runtime, CPU) | Measured (slower than PyTorch CPU here) | `reports/onnx_export.json`, `reports/onnx_verifier_benchmark.json` |
| Mac-local generation (mlx-lm) | Measured | `reports/mac_local_inference_benchmark.json` |
| Mac-local generation (Ollama) | Skipped (no server running) | Runnable once `ollama serve` is started and a model is pulled |
| Mac-local generation (llama.cpp) | Skipped (not configured) | Runnable once `binary_path`/`model_path` are set in `configs/inference_benchmark.yaml` |
| Dense-scoring microbenchmark (Triton/CUDA) | Scaffolded, not run | No CUDA/Triton locally |
| SGLang / MLC / FlashAttention / TVM / MLIR | Architecture notes only | See `docs/inference_runtime_landscape.md` |

## JD alignment

Veritas now includes runtime benchmarking across Transformers fallback, vLLM
endpoint serving, optional ONNX Runtime verifier inference, and optional
Triton dense-scoring kernels.
