# Retrieval Evaluation

- Split: val
- Max queries: 200
- Queries evaluated: 200
- Evidence corpus size: 9804
- Dense backend: sentence-transformers
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Runtime seconds: 154.531

| retriever | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.246 | 0.400 | 0.442 | 0.398 | 0.389 |
| dense | 0.357 | 0.505 | 0.524 | 0.534 | 0.506 |
| hybrid | 0.288 | 0.507 | 0.535 | 0.490 | 0.478 |

## Limitations

- Evaluated on a sample of 200 queries, capped at --max-queries=200.
