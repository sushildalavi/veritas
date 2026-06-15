# Verifier Inference Benchmark

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Git commit: `b8eff12678a1135732f0c32b9cfea751d600aa56`
- torch version: `2.11.0`
- Devices tested: cpu, mps

| device | batch_size | mean_latency_ms | p50_latency_ms | p95_latency_ms | throughput (examples/sec) |
| --- | --- | --- | --- | --- | --- |
| cpu | 1 | 16.02 | 15.784 | 16.912 | 62.42 |
| cpu | 8 | 58.92 | 59.018 | 59.709 | 135.78 |
| cpu | 32 | 208.287 | 208.239 | 209.36 | 153.63 |
| mps | 1 | 6.502 | 6.578 | 7.119 | 153.79 |
| mps | 8 | 23.467 | 23.481 | 23.6 | 340.9 |
| mps | 32 | 87.805 | 87.696 | 88.109 | 364.44 |

- Peak RSS: 850.1 MB
