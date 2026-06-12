# Retrieval Evaluation

- Split: val
- Max queries: 20
- Queries evaluated: 20
- Evidence corpus size: 9804
- Dense backend: sentence-transformers
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Runtime seconds: 68.037

| retriever | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.342 | 0.636 | 0.653 | 0.579 | 0.589 |
| dense | 0.447 | 0.628 | 0.669 | 0.662 | 0.648 |
| hybrid | 0.422 | 0.653 | 0.653 | 0.675 | 0.643 |

## Limitations

- Evaluated on a sample of 20 queries, capped at --max-queries=20.
