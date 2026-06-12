# Ranking Evaluation

- Candidate K: 10
- Learned backend: sklearn-logistic

## fever_train

| strategy | map | mrr | ndcg@5 |
| --- | --- | --- | --- |
| heuristic | 0.556 | 0.556 | 0.500 |
| bm25 | 0.306 | 0.306 | 0.315 |
| dense | 0.100 | 0.100 | 0.000 |
| rrf | 0.133 | 0.133 | 0.000 |
| learned | 0.750 | 0.750 | 0.815 |

## fever_val

| strategy | map | mrr | ndcg@5 |
| --- | --- | --- | --- |
| heuristic | 0.667 | 1.000 | 0.613 |
| bm25 | 1.000 | 1.000 | 1.000 |
| dense | 0.267 | 0.200 | 0.237 |
| rrf | 0.500 | 0.500 | 0.651 |
| learned | 0.417 | 0.333 | 0.571 |

## fever_test

| strategy | map | mrr | ndcg@5 |
| --- | --- | --- | --- |
| heuristic | 1.000 | 1.000 | 1.000 |
| bm25 | 1.000 | 1.000 | 1.000 |
| dense | 1.000 | 1.000 | 1.000 |
| rrf | 1.000 | 1.000 | 1.000 |
| learned | 0.305 | 0.200 | 0.151 |

## scifact_train

| strategy | map | mrr | ndcg@5 |
| --- | --- | --- | --- |
| heuristic | 0.750 | 0.750 | 0.815 |
| bm25 | 0.750 | 0.750 | 0.815 |
| dense | 0.312 | 0.312 | 0.315 |
| rrf | 0.625 | 0.625 | 0.715 |
| learned | 0.238 | 0.238 | 0.250 |

## scifact_val

| strategy | map | mrr | ndcg@5 |
| --- | --- | --- | --- |
| heuristic | 1.000 | 1.000 | 1.000 |
| bm25 | 1.000 | 1.000 | 1.000 |
| dense | 1.000 | 1.000 | 1.000 |
| rrf | 1.000 | 1.000 | 1.000 |
| learned | 0.125 | 0.125 | 0.000 |

## scifact_test

| strategy | map | mrr | ndcg@5 |
| --- | --- | --- | --- |
| heuristic | 0.000 | 0.000 | 0.000 |
| bm25 | 0.000 | 0.000 | 0.000 |
| dense | 0.000 | 0.000 | 0.000 |
| rrf | 0.000 | 0.000 | 0.000 |
| learned | 0.000 | 0.000 | 0.000 |
