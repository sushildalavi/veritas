# Retrieval Evaluation

- Split: val
- Max queries: 20
- Queries evaluated: 20
- Evidence corpus size: 9804
- Dense backend: sentence-transformers
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Runtime seconds: 61.832

| retriever | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.225 | 0.400 | 0.461 | 0.366 | 0.369 |
| dense | 0.381 | 0.528 | 0.558 | 0.575 | 0.524 |
| hybrid | 0.306 | 0.517 | 0.528 | 0.550 | 0.494 |

## Limitations

- Evaluated on a sample of 20 queries, capped at --max-queries=20.
