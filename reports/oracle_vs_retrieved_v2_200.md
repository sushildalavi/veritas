# Oracle vs Retrieved V2

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Git commit: `24ac3ed8a0052eb77cece7d5d717a9ada2867de5`
- Config path: `configs/serving.yaml`
- Dataset source: `structured records from fever_test_large, scifact_test_large`
- Retrieval backend: `bm25_only`
- Reranker backend: `none`
- Split prefixes: `fever_test, scifact_test`
- Suffix: `_large`
- Sample size: 200
- Retrieval runtime (s): 2.344

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
| per_passage_max | oracle | 0.75 | 0.7206 | 0.0 |
| bundle | retrieved | 0.415 | 0.4013 | 0.8 |
| per_passage_max | retrieved | 0.48 | 0.4715 | 0.6133 |

## Delta from oracle

- bundle: accuracy gap=0.375, macro_f1 gap=0.3705
- per_passage_max: accuracy gap=0.27, macro_f1 gap=0.2491
