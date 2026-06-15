# Oracle vs Retrieved V2

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Git commit: `a21ede24d7b0f051091cb5405c50080ad46e0838`
- Config path: `configs/serving.yaml`
- Dataset source: `structured records from fever_test_large, scifact_test_large`
- Retrieval backend: `bm25_only`
- Reranker backend: `none`
- Split prefixes: `fever_test, scifact_test`
- Suffix: `_large`
- Sample size: 100
- Retrieval runtime (s): 1.245

## Retrieval metrics

- recall@1: 0.328
- recall@3: 0.4587
- recall@5: 0.501
- recall@10: 0.5435
- recall@50: 0.5755
- ndcg@5: 0.4751
- ndcg@10: 0.4891

## Verifier metrics

| mode | evidence | accuracy | macro_f1 | nei_false_positive_rate |
| --- | --- | --- | --- | --- |
| bundle | oracle | 0.74 | 0.7419 | 0.0 |
| per_passage_max | oracle | 0.73 | 0.7225 | 0.0 |
| bundle | retrieved | 0.41 | 0.3901 | 0.8065 |
| per_passage_max | retrieved | 0.46 | 0.4498 | 0.6452 |

## Delta from oracle

- bundle: accuracy gap=0.33, macro_f1 gap=0.3518
- per_passage_max: accuracy gap=0.27, macro_f1 gap=0.2727
