# Evidence Pair Dataset Stats

- Corpus size: 9804

## train

- Retriever pairs: 2393
- Reranker pairs: 9572
- Reranker relevance counts: {'positive': 2393, 'negative': 7179}
- Reranker negative type counts: {'bm25_hard_negative': 6995, 'dense_hard_negative': 170, 'random_negative': 14}
- Label distribution (retriever pairs): {'SUPPORTED': 1416, 'REFUTED': 604, 'NOT_ENOUGH_INFO': 373}
- Hard negative count distribution: {'3': 2393}

## val

- Retriever pairs: 532
- Reranker pairs: 2128
- Reranker relevance counts: {'positive': 532, 'negative': 1596}
- Reranker negative type counts: {'bm25_hard_negative': 1563, 'dense_hard_negative': 30, 'random_negative': 3}
- Label distribution (retriever pairs): {'SUPPORTED': 237, 'REFUTED': 176, 'NOT_ENOUGH_INFO': 119}
- Hard negative count distribution: {'3': 532}

## test

- Retriever pairs: 540
- Reranker pairs: 2160
- Reranker relevance counts: {'positive': 540, 'negative': 1620}
- Reranker negative type counts: {'bm25_hard_negative': 1587, 'dense_hard_negative': 31, 'random_negative': 2}
- Label distribution (retriever pairs): {'REFUTED': 195, 'SUPPORTED': 224, 'NOT_ENOUGH_INFO': 121}
- Hard negative count distribution: {'3': 540}

