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

- Re-run the v2 evaluator on larger held-out samples once runtime budget allows.
- Compare BM25-only, hybrid, and cross-encoder reranked profiles under the same evaluator.
- Keep reporting oracle and retrieved numbers side by side; reporting only verifier macro-F1 hides the real ceiling.
