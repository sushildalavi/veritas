# Veritas Architecture Audit

This document maps the current repository structure to the end-to-end verification pipeline.

## Pipeline

1. Data build
   - `scripts/build_large_sample_datasets.py` builds the structured FEVER/SciFact sample.
   - `scripts/build_verifier_dataset.py` converts structured records into clean verifier text examples.
   - `scripts/build_evidence_pair_dataset.py` mines retriever and reranker training pairs with BM25, dense, and random negatives.

2. Retrieval
   - `retrieval/bm25.py` provides lexical retrieval with optional title and metadata-window indexing.
   - `retrieval/dense.py` provides hashing or sentence-transformer retrieval.
   - `retrieval/hybrid.py` fuses lexical, dense, title, and query-expanded candidates with RRF.
   - `scripts/run_retrieval_eval.py` and `scripts/run_retrieval_ablation.py` measure recall and nDCG.

3. Ranking
   - `ranking/reranker.py` contains heuristic and cross-encoder reranking.
   - `scripts/train_cross_encoder_reranker.py` and `scripts/eval_cross_encoder_reranker.py` cover the learned reranker path.

4. Verification
   - `core/evidence_formatting.py` is the single formatting layer that strips marker leakage.
   - `models/deberta_verifier.py` handles sklearn, transformer, and deterministic fallback inference.
   - `models/model_router.py` selects the deployed verifier backend and now exposes per-passage scoring.
   - `scripts/eval_oracle_vs_retrieved_v2.py` separates retrieval ceiling from verifier quality on structured evidence records.

5. Explanation and serving
   - `rag/` builds grounded context and parses explanation outputs.
   - `serving/model_loader.py` wires retrieval, reranking, verifier, and explanation generator backends.
   - `serving/api.py` exposes the FastAPI surface used by the demo and health endpoints.

## Current constraints

- Retrieval remains the dominant ceiling. On the full 650-example v2 report (`reports/oracle_vs_retrieved_v2_full.json`), recall@10 is `0.5334` and the oracle-vs-retrieved per-passage macro-F1 gap is `0.2841`. The earlier 20-example sampled report (recall@10 `0.6667`) was diagnostic only.
- Oracle evidence still materially outperforms retrieved evidence.
- The verifier remains the source of truth for labels; explanation generation is a separate backend.
- The vLLM path is explanation-only and keeps template fallback for local and free environments.

## Obsolete paths removed

- The legacy `scripts/evaluate_oracle_vs_retrieved.py` path was removed after the structured v2 evaluator replaced it.
- Evidence formatting is no longer duplicated across verifier training and inference scripts.
