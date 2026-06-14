# Top-k Evidence Verifier Evaluation

- checkpoint: checkpoints/transformer_verifier_clean
- test_file: data/processed/verifier_test_topk_augmented.jsonl
- sample_size: 642
- headline: oracle macro_f1=0.705, top1=0.450, top3=0.476, top5=0.467, mixed=0.588

| variant | accuracy | macro_f1 | refuted_recall | nei_f1 | accuracy_gap | macro_f1_gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | 0.7134 | 0.7052 | 0.7385 | 0.942 | 0.0 | 0.0 |
| top1 | 0.4611 | 0.4502 | 0.4974 | 0.5331 | 0.2523 | 0.255 |
| top3 | 0.4751 | 0.4756 | 0.559 | 0.507 | 0.2383 | 0.2296 |
| top5 | 0.4673 | 0.4673 | 0.5231 | 0.4541 | 0.2461 | 0.2379 |
| mixed | 0.5903 | 0.5882 | 0.6667 | 0.7336 | 0.1231 | 0.1171 |

## Notes

- top1/top3/top5 use evidence retrieved by BM25 + fine-tuned bi-encoder (RRF fusion), reranked by the fine-tuned cross-encoder reranker, on a held-out test set the verifier was not retrained on.
- top1/top3/top5 all improve macro_f1 over the previous top-1 BM25-only baseline of 0.414 (reports/end_to_end_verifier_eval.json), but remain well below the 0.55 target and far below the oracle macro_f1 of 0.705.
- mixed reaches macro_f1=0.588 but is composed of 50% gold evidence rows (mixed_source == 'gold'), so it does not represent a pure retrieved-evidence result and should not be compared directly against the 0.55 target.
