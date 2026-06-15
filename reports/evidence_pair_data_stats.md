# Evidence Pair Dataset Stats

- Corpus size: 9804

## train

- Positive examples: 2393
- Negative examples: 11965
- Retriever pairs: 2393
- Reranker pairs: 14358
- Reranker relevance counts: {'positive': 2393, 'negative': 11965}
- Reranker negative type counts: {'bm25_hard_negative': 4720, 'dense_hard_negative': 4786, 'random_negative': 2459}
- Label distribution (retriever pairs): {'SUPPORTED': 1416, 'REFUTED': 604, 'NOT_ENOUGH_INFO': 373}
- Hard negative count distribution: {'5': 2393}

## val

- Positive examples: 532
- Negative examples: 2660
- Retriever pairs: 532
- Reranker pairs: 3192
- Reranker relevance counts: {'positive': 532, 'negative': 2660}
- Reranker negative type counts: {'bm25_hard_negative': 1055, 'dense_hard_negative': 1063, 'random_negative': 542}
- Label distribution (retriever pairs): {'SUPPORTED': 237, 'REFUTED': 176, 'NOT_ENOUGH_INFO': 119}
- Hard negative count distribution: {'5': 532}

## test

- Positive examples: 540
- Negative examples: 2700
- Retriever pairs: 540
- Reranker pairs: 3240
- Reranker relevance counts: {'positive': 540, 'negative': 2700}
- Reranker negative type counts: {'bm25_hard_negative': 1071, 'dense_hard_negative': 1080, 'random_negative': 549}
- Label distribution (retriever pairs): {'REFUTED': 195, 'SUPPORTED': 224, 'NOT_ENOUGH_INFO': 121}
- Hard negative count distribution: {'5': 540}

