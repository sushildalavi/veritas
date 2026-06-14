# Bi-Encoder Retriever Evaluation

- split: val
- query_count: 20
- corpus_size: 9804
- runtime_seconds: 111.336

| strategy | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | latency_seconds | backend | model_name |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bm25 | 0.225 | 0.4 | 0.4611 | 0.3663 | 0.3686 | 0.0758 | lexical | bm25 |
| dense_generic | 0.3806 | 0.5278 | 0.5583 | 0.575 | 0.5241 | 0.2612 | semantic | sentence-transformers/all-MiniLM-L6-v2 |
| dense_finetuned | 0.4056 | 0.4667 | 0.4667 | 0.6 | 0.4969 | 0.2582 | semantic | checkpoints/biencoder_retriever |
| hybrid_finetuned | 0.3556 | 0.5111 | 0.5167 | 0.575 | 0.5029 | 0.3095 | hybrid | bm25+checkpoints/biencoder_retriever |

## Notes

- Fine-tuned vs generic dense recall@1: improved by 0.025 (0.3806 -> 0.4056).
- Fine-tuned vs generic dense recall@5: regressed by 0.0611 (0.5278 -> 0.4667).
- Fine-tuned vs generic dense recall@10: regressed by 0.0916 (0.5583 -> 0.4667).
- Fine-tuned vs generic dense mrr: improved by 0.025 (0.575 -> 0.6).
- Fine-tuned vs generic dense ndcg@10: regressed by 0.0272 (0.5241 -> 0.4969).
