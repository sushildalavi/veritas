# Oracle vs Retrieved V2

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Git commit: `36fa254ec914f1ed465e99759b72b7b47e19d5dd`
- Config path: `configs/serving.yaml`
- Dataset source: `structured records from fever_test_large, scifact_test_large`
- Retrieval backend: `bm25_only`
- Reranker backend: `none`
- Split prefixes: `fever_test, scifact_test`
- Suffix: `_large`
- Sample size: 200
- Retrieval runtime (s): 2.214

## Retrieval metrics

- recall@1: 0.2921
- recall@3: 0.3958
- recall@5: 0.4255
- recall@10: 0.4711
- recall@50: 0.5051
- ndcg@5: 0.4139
- ndcg@10: 0.4286

## Verifier metrics

| mode | evidence | accuracy | macro_f1 | nei_false_positive_rate |
| --- | --- | --- | --- | --- |
| bundle | oracle | 0.79 | 0.7718 | 0.0 |
| per_passage_max | oracle | 0.755 | 0.7255 | 0.0 |
| bundle | retrieved | 0.415 | 0.4013 | 0.8 |
| per_passage_max | retrieved | 0.475 | 0.464 | 0.6533 |

## Delta from oracle

- bundle: accuracy gap=0.375, macro_f1 gap=0.3705
- per_passage_max: accuracy gap=0.28, macro_f1 gap=0.2615
