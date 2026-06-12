# Retrieval Evaluation

- Corpus size: 12
- Top K: 5

## fever_train

| retriever | mean_recall@1 | mean_recall@5 | mean_recall@10 | mean_mrr | mean_ndcg@5 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.000 | 0.500 | 0.500 | 0.250 | 0.315 |
| dense | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid | 0.000 | 0.500 | 0.500 | 0.100 | 0.193 |

## fever_val

| retriever | mean_recall@1 | mean_recall@5 | mean_recall@10 | mean_mrr | mean_ndcg@5 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense | 0.000 | 0.500 | 0.500 | 0.200 | 0.237 |
| hybrid | 0.000 | 1.000 | 1.000 | 0.500 | 0.624 |

## fever_test

| retriever | mean_recall@1 | mean_recall@5 | mean_recall@10 | mean_mrr | mean_ndcg@5 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.250 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense | 0.250 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid | 0.250 | 1.000 | 1.000 | 1.000 | 1.000 |

## scifact_train

| retriever | mean_recall@1 | mean_recall@5 | mean_recall@10 | mean_mrr | mean_ndcg@5 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| dense | 0.500 | 1.000 | 1.000 | 0.500 | 0.750 |
| hybrid | 0.500 | 1.000 | 1.000 | 0.667 | 0.750 |

## scifact_val

| retriever | mean_recall@1 | mean_recall@5 | mean_recall@10 | mean_mrr | mean_ndcg@5 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## scifact_test

| retriever | mean_recall@1 | mean_recall@5 | mean_recall@10 | mean_mrr | mean_ndcg@5 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dense | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
