# Final Pareto Analysis

## Frontier

| model | macro_f1 | latency_ms | memory_mb | deployment_feasibility |
| --- | --- | --- | --- | --- |
| distilroberta-clean | 0.711 | 29.7005 | 316.68 | 0.337 |
| sklearn-tfidf-logreg | 0.484 | 0.0167 | 0.60 | 0.999 |

## All measured points

| model | macro_f1 | latency_ms | memory_mb | deployment_feasibility |
| --- | --- | --- | --- | --- |
| distilroberta-clean | 0.711 | 29.7005 | 316.68 | 0.337 |
| sklearn-tfidf-logreg | 0.484 | 0.0167 | 0.60 | 0.999 |

## Not trained (blocked)

| model | status | note |
| --- | --- | --- |
| qlora-tinyllama | not trained - GPU required | see reports/qlora_BLOCKED_GPU_REQUIRED.md |
| dpo-tinyllama | not trained - QLoRA required | see reports/dpo_BLOCKED_QLORA_REQUIRED.md |

