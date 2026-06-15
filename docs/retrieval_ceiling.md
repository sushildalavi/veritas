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

## Negative results: retrieval profile and threshold calibration (full 650-set)

Two follow-up experiments were run on the full 650-example set
(`reports/retrieval_profile_comparison_650.md`, `reports/threshold_calibration_650.md`) and
neither beat the current `bm25_only` + `support_threshold=0.55` / `refute_threshold=0.5` defaults:

- `hybrid_bm25_sentence_transformer` improved retrieval recall@10 (`0.5334` -> `0.5714`) and
  nDCG@10 (`0.4816` -> `0.5234`), but retrieved per-passage macro-F1 went down (`0.3887` ->
  `0.3776`) and the oracle gap widened slightly (`0.2841` -> `0.2952`), at more than double the
  runtime.
- `hybrid_bm25_dense` did not beat `bm25_only` on recall@10, nDCG@10, or retrieved macro-F1.
- A support/refute/margin grid search on a calibration split (`fever_val_large` +
  `scifact_val_large`, 650 examples) found the current defaults (`0.55` / `0.5` / `0.0`) already
  optimal on the tested grid; held-out macro-F1, accuracy, and REFUTED prediction rate were
  unchanged.

Together these indicate the next bottleneck is **verifier robustness under noisy retrieved
evidence**, not retrieval depth or simple threshold tuning -- better retrieval (the
sentence-transformer hybrid) did not translate into a better verifier verdict, and threshold
recalibration had nothing left to gain.

## Analysis-only: retrieved-evidence verifier robustness dataset

To investigate the verifier-robustness hypothesis above, a new per-passage diagnostic dataset
was built (`scripts/build_retrieved_evidence_dataset.py`,
`reports/retrieved_evidence_dataset_stats.md`). Each example is a single (claim, passage) pair
labeled `SUPPORTS` / `REFUTES` / `NEI`, covering both gold (`positive_oracle`) and retrieved
(`positive_retrieved`) positives plus four hard-negative categories
(`same_entity_insufficient`, `same_topic_missing_fact`, `near_miss`, `irrelevant`), with a
claim/passage lexical-overlap score stored per pair as an inference-time relevance signal.
The calibration split (650 claims, `fever_val_large` + `scifact_val_large`) has 3663 pairs;
the holdout split (650 claims, `fever_test_large` + `scifact_test_large`) has 3676 pairs.
Both splits are roughly 75% `NEI` pairs, reflecting that most retrieved passages are not gold
evidence.

A new eval (`scripts/eval_retrieved_evidence_verifier.py`,
`reports/retrieved_evidence_verifier_eval.md`) compares the verifier's current per-passage
prediction against the same prediction forced to `NEI` whenever the lexical-overlap score is
below a threshold tuned on the calibration split (best threshold: `0.5`). On the holdout split,
the relevance-gated predictor improves per-passage macro-F1 from `0.3192` to `0.4511`
(+0.1319), driven almost entirely by `NEI` F1 (`0.395` -> `0.8196`); `SUPPORTS`/`REFUTES` F1
hold roughly flat or drop slightly. The gain holds on both FEVER (`0.3541` -> `0.4752`) and
SciFact (`0.2234` -> `0.3734`).

**This is a per-passage diagnostic result, not a full-650-set improvement.** The official
oracle-vs-retrieved metric (`reports/oracle_vs_retrieved_v2_full.json`, retrieved macro-F1
`0.3887`) is unaffected by this experiment -- the gate has not been integrated into
`ModelRouter`, and this benchmark's heavy `NEI` skew means a gate that defaults to `NEI` scores
well here without yet being validated against the claim-level REFUTED-overprediction problem
documented above. A reasonable next step, if pursued, would be to prototype the lexical-overlap
gate inside `ModelRouter.predict` (forcing `NOT ENOUGH INFO` when no retrieved passage clears
the relevance threshold) and re-run `scripts/eval_oracle_vs_retrieved_v2.py` on the full 650-set
to check whether it actually reduces REFUTED overprediction and improves retrieved macro-F1
end-to-end -- no such claim is made here.
