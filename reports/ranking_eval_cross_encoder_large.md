# Ranking Evaluation

- Split: val
- Max queries: 200
- Train query cap: 40
- Queries evaluated: 200
- Candidate K: 10
- Learned backend: sklearn-logistic
- Cross encoder enabled: True
- Cross encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Runtime seconds: 73.648

| strategy | map | mrr | ndcg@5 | ndcg@10 | recall@5 |
| --- | --- | --- | --- | --- | --- |
| heuristic | 0.368 | 0.386 | 0.367 | 0.402 | 0.421 |
| bm25 | 0.376 | 0.406 | 0.379 | 0.387 | 0.404 |
| dense | 0.123 | 0.115 | 0.077 | 0.085 | 0.089 |
| rrf | 0.242 | 0.265 | 0.222 | 0.271 | 0.259 |
| learned | 0.498 | 0.508 | 0.505 | 0.533 | 0.541 |
| cross_encoder | 0.540 | 0.562 | 0.549 | 0.564 | 0.567 |
| cross_encoder_plus_learned | 0.584 | 0.587 | 0.597 | 0.604 | 0.623 |

## Limitations

- Evaluated on a sample of 200 queries, capped at --max-queries=200.
- Learned ranking remains sample-scale and depends on the sampled evidence corpus.
