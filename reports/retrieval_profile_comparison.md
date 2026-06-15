# Retrieval Profile Comparison

- Config path: `configs/serving.yaml`
- Split prefixes: `fever_test, scifact_test`
- Max examples: 50

| profile | status | retrieval_backend | reranker_backend | recall@10 | ndcg@10 | per_passage_macro_f1 | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_only | measured | bm25_only | none | 0.601 | 0.5678 | 0.3693 | 15.272 |
| dense_only | measured | dense_only | none | 0.1507 | 0.0801 | 0.281 | 18.045 |
| hybrid_bm25_dense | measured | bm25_hashing_hybrid | none | 0.5713 | 0.5168 | 0.4748 | 17.721 |
| hybrid_with_query_expansion | measured | bm25_hashing_hybrid | none | 0.597 | 0.5796 | 0.4124 | 19.229 |
| hybrid_with_reranker | measured | bm25_hashing_hybrid | cross_encoder | 0.621 | 0.6083 | 0.3976 | 40.337 |
| hybrid_bm25_sentence_transformer | measured | bm25_sentence_transformer_hybrid | none | 0.6377 | 0.6106 | 0.4595 | 75.7 |

## Skip notes

