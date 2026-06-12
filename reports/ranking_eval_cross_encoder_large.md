# Ranking Evaluation

- Split: val
- Max queries: 2
- Train query cap: 40
- Queries evaluated: 2
- Candidate K: 2
- Learned backend: sklearn-logistic
- Cross encoder enabled: True
- Cross encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Runtime seconds: 23.884

| strategy | map | mrr | ndcg@5 | ndcg@10 | recall@5 |
| --- | --- | --- | --- | --- | --- |
| heuristic | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| bm25 | 0.350 | 0.500 | 0.425 | 0.425 | 0.500 |
| dense | 0.208 | 0.167 | 0.285 | 0.285 | 0.500 |
| rrf | 0.350 | 0.500 | 0.425 | 0.425 | 0.500 |
| learned | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| cross_encoder | 0.417 | 0.500 | 0.460 | 0.460 | 0.500 |
| cross_encoder_plus_learned | 0.417 | 0.500 | 0.460 | 0.460 | 0.500 |

## Limitations

- Evaluated on a sample of 2 queries, capped at --max-queries=2.
- Learned ranking remains sample-scale and depends on the sampled evidence corpus.
