# Inference Serving Benchmark

- Git commit: `b8eff12678a1135732f0c32b9cfea751d600aa56`
- torch version: `2.11.0`

## transformers backend (local generation)

- Status: measured
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Devices tested: cpu, mps
- max_new_tokens: 32

| device | mean_latency_ms | mean_generated_tokens | tokens/sec |
| --- | --- | --- | --- |
| cpu | 1978.23 | 32 | 16.18 |
| mps | 1048.07 | 32 | 30.53 |

## vLLM backend

- Status: skipped
- vLLM base URL: `http://127.0.0.1:8001`
- vLLM model: `Qwen/Qwen2.5-1.5B-Instruct`
- Reason: vllm endpoint unavailable in current environment
- Next steps: Start a vLLM server (see docs/vllm_serving.md / scripts/serve_vllm_explanations.py), then re-run scripts/benchmark_inference_serving.py.
