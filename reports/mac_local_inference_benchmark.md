# Mac-Local Inference Benchmark

- Git commit: `cf6a4228e155f330aaed6bb4561f00d2e5e8ed87`
- Timestamp: 2026-06-15T21:27:49.359895+00:00

## mlx-lm

- Status: measured
- Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Device: mps
- max_tokens: 32

| mean_latency_ms | p50_latency_ms | p95_latency_ms | tokens/sec | requests/sec |
| --- | --- | --- | --- | --- |
| 595.85 | 582.66 | 621.71 | 53.7 | 1.678 |

## Ollama

- Status: skipped
- Reason: ollama endpoint http://localhost:11434 is unreachable in this environment
- Next steps: Start ollama (`ollama serve`), pull a model (`ollama pull qwen2.5:1.5b`), then re-run scripts/benchmark_mac_local_inference.py.

## llama.cpp

- Status: skipped
- Reason: mac_local_backends.llama_cpp.binary_path / .model_path are not configured in configs/inference_benchmark.yaml
- Next steps: Build llama.cpp with Metal support (LLAMA_METAL=1), download a GGUF model, set mac_local_backends.llama_cpp.binary_path and .model_path in configs/inference_benchmark.yaml, then re-run scripts/benchmark_mac_local_inference.py.
