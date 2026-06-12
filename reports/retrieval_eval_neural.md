# Retrieval Evaluation

- Split: val
- Max queries: 5
- Queries evaluated: 2
- Evidence corpus size: 12
- Dense backend: sentence-transformers
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Runtime seconds: 14.854

| retriever | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid | 0.000 | 1.000 | 1.000 | 0.500 | 0.662 |

## Limitations

- Evaluated on a sample of 2 queries, capped at --max-queries=5.
