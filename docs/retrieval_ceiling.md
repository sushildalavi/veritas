# Retrieval Ceiling Notes

The verifier is no longer the only bottleneck in Veritas. The structured v2 evaluation makes the gap visible directly.

## Sampled v2 measurements

Measured from `reports/oracle_vs_retrieved_v2.json` after threshold calibration on the BM25 serving profile with `20` sampled held-out records:

- Retrieval recall@1: `0.5167`
- Retrieval recall@5: `0.6417`
- Retrieval recall@10: `0.6667`
- Retrieval nDCG@10: `0.6344`
- Oracle verifier macro-F1 (per-passage): `0.7086`
- Retrieved verifier macro-F1 (per-passage): `0.5000`
- Retrieved verifier macro-F1 (bundled evidence): `0.4205`

## Interpretation

- Per-passage scoring helps, but it does not close the retrieval gap by itself.
- Even with deeper candidate pools and calibrated thresholds, missing gold evidence still caps end-to-end quality.
- Retrieval recall@10 around `0.67` means roughly one-third of relevant evidence remains outside the candidate set on this slice.

## Practical next steps

- Done: the v2 evaluator has been re-run on larger held-out samples, up to the full 650-example
  `fever_test_large` + `scifact_test_large` test set (`reports/oracle_vs_retrieved_v2_full.md`).
  Oracle per-passage macro-F1 is `0.6728`, retrieved is `0.3887`, recall@10 is `0.5334` -- a gap
  of `0.2841`, consistent with the 20/100/200-example slices.
- Done: BM25-only, dense, hybrid, query-expansion, cross-encoder reranked, and real MiniLM-hybrid
  profiles have been compared under the same evaluator (`reports/retrieval_profile_comparison.md`).
- Done: a failure-mode breakdown of the full 650-example run (`reports/error_analysis_650.md`,
  `reports/error_analysis_650.json`) decomposes the gap further. The largest single bucket (34%,
  222/650) is `oracle_correct_retrieved_wrong` -- cases where the verifier would have been correct
  given gold evidence but retrieved evidence flipped the verdict, ahead of `oracle_wrong_retrieved_wrong`
  (25%, 164/650), where the verifier is wrong even with gold evidence. The retrieved-evidence verifier
  is also heavily REFUTED-biased (REFUTED predicted for 371/650 examples vs. 200 gold REFUTED;
  precision 0.38, recall 0.71), and SciFact retrieved macro-F1 (0.18) is far below FEVER's (0.44)
  despite SciFact having higher recall@10 (0.63 vs 0.51) -- evidence of a dataset-specific verifier
  weakness on top of the shared retrieval ceiling.
- Keep reporting oracle and retrieved numbers side by side; reporting only verifier macro-F1 hides the real ceiling.
