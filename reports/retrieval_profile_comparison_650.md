# Retrieval Profile Comparison (Full 650-Example Set)

- Config path: `configs/serving.yaml`
- Split prefixes: `fever_test, scifact_test`
- Suffix: `_large`
- Sample size: 650

## Retrieval metrics

| profile | recall@1 | recall@3 | recall@5 | recall@10 | recall@50 | nDCG@5 | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_only | 0.3559 | 0.4586 | 0.4979 | 0.5334 | 0.5706 | 0.4708 | 0.4816 |
| hybrid_bm25_dense | 0.2923 | 0.4228 | 0.4613 | 0.5113 | 0.5398 | 0.4134 | 0.4288 |
| hybrid_bm25_sentence_transformer | 0.3903 | 0.5027 | 0.5416 | 0.5714 | 0.5938 | 0.5166 | 0.5234 |

## Retrieved per-passage verifier metrics

| profile | accuracy | macro_f1 | SUPPORTED F1 | REFUTED F1 | NEI F1 | refuted_pred_rate | nei_fp_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_only | 0.4062 | 0.3887 | 0.3211 | 0.4974 | 0.3476 | 0.5708 | 0.7098 |
| hybrid_bm25_dense | 0.4062 | 0.3864 | 0.3 | 0.5106 | 0.3485 | 0.56 | 0.692 |
| hybrid_bm25_sentence_transformer | 0.3969 | 0.3776 | 0.2986 | 0.4974 | 0.3369 | 0.5708 | 0.7188 |

## Oracle gap and runtime

| profile | oracle macro_f1 | retrieved macro_f1 | macro_f1 gap | total runtime (s) |
| --- | --- | --- | --- | --- |
| bm25_only | 0.6728 | 0.3887 | 0.2841 | 177.664 |
| hybrid_bm25_dense | 0.6728 | 0.3864 | 0.2864 | 185.537 |
| hybrid_bm25_sentence_transformer | 0.6728 | 0.3776 | 0.2952 | 383.596 |
