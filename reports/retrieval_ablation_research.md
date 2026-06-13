# Retrieval Ablation

- split: val
- query_count: 20
- corpus_size: 9804
- runtime_seconds: 56.038

| strategy | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | latency_seconds | backend | model_name | memory_note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| bm25 | 0.225 | 0.4 | 0.4611 | 0.3663 | 0.3686 | 0.0581 | lexical | bm25_only | Minimal incremental memory; no neural weights loaded. |
| dense_mini | 0.3806 | 0.5278 | 0.5583 | 0.575 | 0.5241 | 0.2334 | semantic | sentence-transformers/all-MiniLM-L6-v2 | Moderate memory footprint (~80MB model weights plus embeddings). |
| hybrid_mini | 0.3056 | 0.5167 | 0.5278 | 0.55 | 0.4939 | 0.2831 | hybrid | bm25+all-MiniLM-L6-v2 | Moderate memory footprint (~80MB model weights plus embeddings). |

## Notes

- BGE-M3 dense retrieval attempted but unavailable: RuntimeError: Invalid buffer size: 9.52 GiB
