# Oracle vs Retrieved V2

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Retrieval backend: `bm25_only`
- Reranker backend: `none`
- Split prefixes: `fever_test, scifact_test`
- Suffix: `_large`
- Sample size: 20
- Retrieval runtime (s): 0.245

## Retrieval metrics

- recall@1: 0.5167
- recall@3: 0.6167
- recall@5: 0.6417
- recall@10: 0.6667
- recall@50: 0.7167
- ndcg@5: 0.621
- ndcg@10: 0.6344

## Verifier metrics

| mode | evidence | accuracy | macro_f1 | nei_false_positive_rate |
| --- | --- | --- | --- | --- |
| bundle | oracle | 0.7 | 0.7086 | 0.0 |
| per_passage_max | oracle | 0.7 | 0.7086 | 0.0 |
| bundle | retrieved | 0.55 | 0.4205 | 1.0 |
| per_passage_max | retrieved | 0.55 | 0.4873 | 0.75 |

## Delta from oracle

- bundle: accuracy gap=0.15, macro_f1 gap=0.2881
- per_passage_max: accuracy gap=0.15, macro_f1 gap=0.2213
