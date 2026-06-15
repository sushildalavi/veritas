# Oracle vs Retrieved V2

- Checkpoint: `checkpoints/transformer_verifier_robust`
- Git commit: `a9a721d2faa0990707a1113de0cdbf7d178c5e86`
- Config path: `configs/serving.yaml`
- Dataset source: `structured records from fever_test_large, scifact_test_large`
- Retrieval backend: `bm25_only`
- Reranker backend: `none`
- Split prefixes: `fever_test, scifact_test`
- Suffix: `_large`
- Sample size: 650
- Retrieval runtime (s): 7.693

## Retrieval metrics

- recall@1: 0.3559
- recall@3: 0.4586
- recall@5: 0.4979
- recall@10: 0.5334
- recall@50: 0.5706
- ndcg@5: 0.4708
- ndcg@10: 0.4816

## Verifier metrics

| mode | evidence | accuracy | macro_f1 | nei_false_positive_rate |
| --- | --- | --- | --- | --- |
| bundle | oracle | 0.6185 | 0.49 | 0.0134 |
| per_passage_max | oracle | 0.5615 | 0.4376 | 0.0134 |
| bundle | retrieved | 0.3738 | 0.2802 | 0.2232 |
| per_passage_max | retrieved | 0.3862 | 0.2829 | 0.1518 |

## Delta from oracle

- bundle: accuracy gap=0.2447, macro_f1 gap=0.2098
- per_passage_max: accuracy gap=0.1753, macro_f1 gap=0.1547
