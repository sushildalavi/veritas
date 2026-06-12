# Ranking Evaluation

- Split: val
- Max queries: 2
- Train query cap: 40
- Queries evaluated: 2
- Candidate K: 5
- Learned backend: sklearn-logistic
- Cross encoder enabled: True
- Cross encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Runtime seconds: 41.593

| strategy | map | mrr | ndcg@5 | ndcg@10 | recall@5 |
| --- | --- | --- | --- | --- | --- |
| heuristic | 0.183 | 0.167 | 0.272 | 0.272 | 0.500 |
| bm25 | 0.188 | 0.250 | 0.193 | 0.290 | 0.250 |
| dense | 0.091 | 0.071 | 0.000 | 0.194 | 0.000 |
| rrf | 0.113 | 0.125 | 0.132 | 0.221 | 0.250 |
| learned | 0.417 | 0.500 | 0.460 | 0.460 | 0.500 |
| cross_encoder | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| cross_encoder_plus_learned | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

## Limitations

- Evaluated on a sample of 2 queries, capped at --max-queries=2.
- Learned ranking remains sample-scale and depends on the sampled evidence corpus.
