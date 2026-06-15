# Error Analysis: Oracle vs Retrieved V2 (650-example full set)

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Git commit: `9246750c5450024958da312637f61dc42f04da0c`
- Config path: `configs/serving.yaml`
- Dataset source: `structured records from fever_test_large, scifact_test_large`
- Retrieval backend: `bm25_only`
- Reranker backend: `none`
- Sample size: 650
- support_threshold: 0.55, refute_threshold: 0.5
- Label distribution: {'SUPPORTED': 226, 'REFUTED': 200, 'NOT ENOUGH INFO': 224}

## 1-2. Retrieved per_passage_max: confusion matrix and per-class metrics

- Accuracy: 0.4062
- Macro F1: 0.3887

Confusion matrix (rows=gold, cols=predicted, order=['SUPPORTED', 'REFUTED', 'NOT ENOUGH INFO']):

- SUPPORTED: [57, 121, 48]
- REFUTED: [21, 142, 37]
- NOT ENOUGH INFO: [51, 108, 65]

| label | precision | recall | f1 |
| --- | --- | --- | --- |
| SUPPORTED | 0.4419 | 0.2522 | 0.3211 |
| REFUTED | 0.3827 | 0.71 | 0.4974 |
| NOT ENOUGH INFO | 0.4333 | 0.2902 | 0.3476 |

## 3. Retrieval-hit vs retrieval-miss breakdown

| gold evidence in | count | hit_rate | accuracy | macro_f1 |
| --- | --- | --- | --- | --- |
| top-10 | 374 | 0.5754 | 0.4786 | 0.3311 |
| top-50 | 396 | 0.6092 | 0.4798 | 0.3338 |

_Misses: 276 examples missing gold evidence in top-10, 254 missing in top-50._

## 4. Oracle vs retrieved agreement (per_passage_max)

| outcome | count | rate |
| --- | --- | --- |
| oracle_correct_retrieved_correct | 232 | 0.3569 |
| oracle_correct_retrieved_wrong | 222 | 0.3415 |
| oracle_wrong_retrieved_wrong | 164 | 0.2523 |
| oracle_wrong_retrieved_correct | 32 | 0.0492 |

## 5. Per-dataset breakdown

| dataset | count | label_distribution | recall@10 | recall@50 | ndcg@10 | oracle acc/F1 | retrieved acc/F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fever | 500 | {'SUPPORTED': 162, 'REFUTED': 163, 'NOT ENOUGH INFO': 175} | 0.5055 | 0.5458 | 0.4611 | 0.714/0.6879 | 0.448/0.4392 |
| scifact | 150 | {'SUPPORTED': 64, 'REFUTED': 37, 'NOT ENOUGH INFO': 49} | 0.6267 | 0.6533 | 0.55 | 0.6467/0.6289 | 0.2667/0.1817 |

## 6. Retrieval profile comparison (50-example slice)

- Source: `reports/retrieval_profile_comparison.json` (max_examples=50)
- Best recall@10: `hybrid_bm25_sentence_transformer` (0.6377)
- Best retrieved per-passage macro-F1: `hybrid_bm25_dense` (0.4748)
- bm25_only baseline: recall@10=0.601, per_passage_macro_f1=0.3693

| profile vs bm25_only | recall@10 delta | per_passage_macro_f1 delta |
| --- | --- | --- |
| hybrid_bm25_dense | -0.0297 | 0.1055 |
| hybrid_with_query_expansion | -0.004 | 0.0431 |
| hybrid_with_reranker | 0.02 | 0.0283 |
| hybrid_bm25_sentence_transformer | 0.0367 | 0.0902 |

## 7-8. Failure examples

- 40 failure examples saved to `reports/failure_examples_650.jsonl` (claim, gold/predicted/oracle/retrieved labels, retrieved evidence texts, retrieval hit flags, class scores).

