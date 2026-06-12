# Ranking Evaluation

- Split: val
- Max queries: 2
- Queries evaluated: 2
- Candidate K: 10
- Learned backend: sklearn-logistic
- Cross encoder enabled: True
- Cross encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Runtime seconds: 11.813

| strategy | map | mrr | ndcg@5 | ndcg@10 | recall@5 |
| --- | --- | --- | --- | --- | --- |
| heuristic | 0.833 | 1.000 | 0.807 | 0.916 | 0.750 |
| bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense | 0.633 | 0.600 | 0.619 | 0.728 | 0.750 |
| rrf | 0.750 | 0.750 | 0.825 | 0.825 | 1.000 |
| learned | 0.271 | 0.229 | 0.285 | 0.443 | 0.500 |
| cross_encoder | 0.667 | 0.667 | 0.750 | 0.750 | 1.000 |
| cross_encoder_plus_learned | 0.562 | 0.562 | 0.500 | 0.658 | 0.500 |

## Limitations

- Evaluated on a sample of 2 queries, capped at --max-queries=2.
- Learned ranking remains sample-scale and depends on the sampled evidence corpus.
