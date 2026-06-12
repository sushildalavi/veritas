# Retrieval Evaluation

- Split: val
- Max queries: 20
- Queries evaluated: 2
- Evidence corpus size: 12
- Dense backend: hashing
- Embedding model: hashing
- Runtime seconds: 0.003

| retriever | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense | 0.500 | 0.750 | 1.000 | 0.600 | 0.728 |
| hybrid | 0.000 | 1.000 | 1.000 | 0.417 | 0.587 |

## Limitations

- Evaluated on a sample of 2 queries, capped at --max-queries=20.
- Dense retrieval used the hashing backend, which is a lightweight baseline.
