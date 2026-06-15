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
  Oracle per-passage macro-F1 is `0.6748`, retrieved is `0.3833`, recall@10 is `0.5334` -- a gap
  of `0.2915`, consistent with the 20/100/200-example slices.
- Done: BM25-only, dense, hybrid, query-expansion, cross-encoder reranked, and real MiniLM-hybrid
  profiles have been compared under the same evaluator (`reports/retrieval_profile_comparison.md`).
- Keep reporting oracle and retrieved numbers side by side; reporting only verifier macro-F1 hides the real ceiling.
