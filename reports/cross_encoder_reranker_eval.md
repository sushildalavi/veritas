# Cross-Encoder Reranker Evaluation

- split: val
- num_queries: 20
- candidate_k: 10
- evidence_corpus_size: 9804
- generic_model: cross-encoder/ms-marco-MiniLM-L-6-v2
- checkpoint: checkpoints/cross_encoder_reranker
- runtime_seconds: 50.134

| strategy | map | mrr | ndcg@5 | ndcg@10 | recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rrf | 0.2025 | 0.2152 | 0.1775 | 0.2397 | 0.25 |
| learned | 0.4583 | 0.4318 | 0.4729 | 0.498 | 0.5667 |
| cross_encoder_generic | 0.6033 | 0.6536 | 0.6073 | 0.6038 | 0.5778 |
| cross_encoder_finetuned | 0.5313 | 0.5792 | 0.5607 | 0.5689 | 0.5972 |
| cross_encoder_finetuned_plus_learned | 0.5697 | 0.5917 | 0.5735 | 0.6017 | 0.5778 |

## Notes

- Fine-tuned vs generic cross-encoder map: regressed by 0.072 (0.6032738095238095 -> 0.5312691910486028).
- Fine-tuned vs generic cross-encoder mrr: regressed by 0.0744 (0.6535714285714286 -> 0.5791666666666667).
- Fine-tuned vs generic cross-encoder ndcg@5: regressed by 0.0466 (0.6073007587339552 -> 0.560732182270487).
- Fine-tuned vs generic cross-encoder ndcg@10: regressed by 0.0349 (0.6037629706979012 -> 0.5688795667599646).
- Fine-tuned vs generic cross-encoder recall@5: improved by 0.0194 (0.5777777777777777 -> 0.5972222222222222).
