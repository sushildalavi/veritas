# Veritas Repo Audit & Cleanup Plan

Generated before any deletions. Safety branch `pre-cleanup-snapshot` already exists
(off `main` HEAD, before this audit). All proposed removals of git-tracked files would
happen via normal commits on `main` (recoverable from history). Untracked/gitignored
build artifacts (checkpoints, `__pycache__`, etc.) are not in git history at all, so
removing them is just disk cleanup, not a history operation.

Legend: **1=keep/load-bearing, 2=redundant/generated, 3=obsolete experiment, 4=uncertain/manual review**

---

## A. In-progress Phase 6 work (uncommitted) — category 4

This is the most recent work-in-progress, currently paused mid-fix.

| Item | Class | Notes |
| --- | --- | --- |
| `configs/retrieved_augmented_verifier.yaml` | 4 | New config for the retrieved-augmented verifier training run. |
| `scripts/train_retrieved_augmented_verifier.py` | 4 | New script, includes the fix for the NEI shortcut-learning bug (two hard-negative types). Not yet validated by a successful run. |
| `scripts/train_transformer_verifier_clean.py` (uncommitted diff) | 4 | Small additive change (`callbacks` param on `_build_trainer` for `EarlyStoppingCallback`). Only needed by the script above. Safe/non-breaking on its own. |
| `reports/retrieved_augmented_verifier_eval.{json,md}` | 2 | **From the FIRST (buggy) training run** — shortcut-learning collapse (top1/3/5 macro-F1 ~0.36, oracle inflated to 0.771 by a spurious `[E1]`-prefix cue). Misleading if left as-is. |
| `checkpoints/retrieved_augmented_verifier/` (1.2GB, untracked) | 2 | Checkpoint from that same buggy run. |

**Recommendation:** depends on the "new approach" decision (Section K). If Phase 6 continues:
keep the config/script/diff, delete the stale buggy report + checkpoint, rerun later.
If Phase 6 is abandoned for the new approach: delete all five items (including reverting
the `train_transformer_verifier_clean.py` diff, since nothing else uses `callbacks`).

---

## B. Checkpoints on disk (3.2GB total, all gitignored / untracked except `checkpoints/verifier`)

| Checkpoint | Size | Class | Notes |
| --- | --- | --- | --- |
| `checkpoints/biencoder_retriever/` | 87M | 1 | Phase 3, committed eval report references it. |
| `checkpoints/cross_encoder_reranker/` | 87M | 1 | Phase 4, committed eval report references it. |
| `checkpoints/transformer_verifier_clean/` | 317M | 1 | Current default verifier (README, configs, serving). |
| `checkpoints/deberta_verifier_clean/` | 1.1G | 1 | Current challenger, referenced in README/final_results. |
| `checkpoints/verifier_clean/` | 620K | 1 | Current sklearn fallback (README, configs, serving). |
| `checkpoints/mlx_lora_verifier/` | 20M | 1 | Explanation adapter, referenced in final_results/configs. |
| `checkpoints/mlx_lora_verifier_300/` | 40M | 1 | 300-step variant compared in `mlx_lora_comparison.json` (part of final_results narrative). |
| `checkpoints/verifier/` (tracked: `model.joblib` 620KB + `metadata.json`) | 20K (git) | 4 | **Old** sklearn checkpoint from `scripts/train_verifier.py`. Distinct from `checkpoints/verifier_clean/`. Still the `core/config.py` default `sklearn_checkpoint` and the `make train-verifier` target, but README/docs/serving configs all point at `verifier_clean` instead. Looks like a leftover from before the "clean" rewrite. |
| `checkpoints/transformer_verifier/` | 317M | 4 | **Old** verifier from `scripts/train_transformer_verifier.py` (pre-"clean" rewrite). Only kept alive by `core/config.py`'s legacy fallback chain and `tests/test_verifier_checkpoint.py`/`tests/test_config.py` (which test the fallback *path string*, not the actual checkpoint contents — no test loads this checkpoint). |
| `checkpoints/retrieved_augmented_verifier/` | 1.2G | 2 | See Section A. |

**Recommendation:** B's "1" rows stay untouched. The two "4" rows (`checkpoints/verifier`,
`checkpoints/transformer_verifier`, ~317MB + 20KB) are a legacy pre-"clean" generation that the
current docs/serving no longer point at — candidates for removal once Section C is resolved
(removing the checkpoint without removing the script/config that produces it would just let it
regenerate later, so these are coupled decisions).

---

## C. Duplicate "old vs clean/final" script generations — category 4

The repo has several `X.py` / `X_clean.py` / `X_final.py` pairs where the newer one is what's
actually documented and wired into the "final" narrative, but the older one is still reachable
from the Makefile or tests.

| Old script | New script | Old script's role today | Recommendation |
| --- | --- | --- | --- |
| `scripts/train_verifier.py` → `checkpoints/verifier` | `scripts/train_verifier_clean.py` → `checkpoints/verifier_clean` | `make train-verifier` / `train-verifier-smoke` targets; `core/config.py` default `sklearn_checkpoint`. | Either retire `train_verifier.py` + repoint Makefile/config default to `verifier_clean`, or keep both if you want the unweighted baseline preserved for comparison. **Needs your call** — touches `core/config.py` defaults. |
| `scripts/train_transformer_verifier.py` → `checkpoints/transformer_verifier` | `scripts/train_transformer_verifier_clean.py` → `checkpoints/transformer_verifier_clean` | Only `tests/test_verifier_checkpoint.py` (`_to_markdown` import) and `core/config.py` legacy fallback path. | `_to_markdown` exists in both files with the same signature — the test could import from `_clean` instead, letting the old script + checkpoint + legacy fallback path be removed. **Needs your call** — touches `core/config.py` fallback chain (small risk). |
| `scripts/eval_faithfulness.py` → `reports/faithfulness_eval.json` | `scripts/eval_faithfulness_final.py` → `reports/faithfulness_comparison.json` | `make eval-faithfulness` / `make all-evals`. `faithfulness_eval.json` is **not** referenced by `final_results.json`. | `faithfulness_comparison.json` (via `run_final_evaluation_suite.py`) is the one that feeds the final report. The non-`_final` script/report look like an earlier draft. **Needs your call** — low risk either way, just Makefile wiring. |
| `scripts/pareto_analysis.py` → `reports/pareto_analysis.json` | `scripts/pareto_analysis_final.py` → `reports/final_pareto_analysis.json` | `make pareto-analysis` / `make all-evals`. `pareto_analysis.json` is **not** referenced by `final_results.json`. | Same pattern as above. **Needs your call.** |

These are all "still technically wired" so I'm not calling them dead code, but they're the kind
of duplication the "start fresh" framing is probably aiming at. None of this needs to block the
rest of the cleanup — it can be its own small follow-up commit/decision.

---

## D. Dead scaffolding (never wired up) — category 3

| Item | Notes |
| --- | --- |
| `training/train_deberta.py`, `training/train_ranker.py`, `training/train_roberta.py` | Phase-1-era stubs (24–56 lines each) that just `print()` a loaded config — never implement actual training. Fully superseded by `scripts/train_*.py`. Only "used" via `tests/test_imports.py` (`import training` package import). |
| `models/roberta_baseline.py` (`RobertaBaselineVerifier`) | Exported from `models/__init__.py` but never used by `ModelRouter` or anywhere else. |
| `data/build_preference_pairs.py` | Superseded by `scripts/build_preference_pairs_real.py` (which produces the actual `data/processed/preference_pairs.jsonl` used in `final_results.json`). No references anywhere except itself. |

**Recommendation:** safe to remove. `tests/test_imports.py` would need `training` dropped from
its import list (or the package emptied to just `__init__.py` + `config.py`/`losses.py` if those
are still used by the stubs' replacements — quick check needed before deleting `training/config.py`
and `training/losses.py`, since `label_to_index` etc. might be used elsewhere).

---

## E. `docs/archive/` (CUDA/QLoRA/DPO experiments, ~132KB total) — category 1 (as currently scoped)

`PROJECT_SUMMARY.md` already documents these as a **deliberately preserved historical archive**
("Neither is part of the final architecture... Legacy CUDA/Colab/Kaggle experiments are preserved
under `docs/archive/...`"). This was an explicit decision from the prior cleanup (commit `0b59fcd`).
Small footprint, already isolated from the active codebase, doesn't interfere with anything.

**Recommendation:** keep as-is unless the "new approach" specifically wants a from-scratch repo
with no historical baggage — flagging this as a decision point, but I'd lean toward keeping it
(it's cheap, isolated, and documents why CUDA/DPO/QLoRA were ruled out).

---

## F. MLX LoRA explanation sub-pipeline — category 1 (mostly)

`scripts/build_explanation_sft_dataset.py`, `train_mlx_lora_explanation.py`,
`eval_mlx_lora_explanation.py`, `eval_preference_reranking.py`, `compare_mlx_lora_adapters.py`,
`configs/mlx_lora_explanation_qwen15b.yaml`, `data/processed/explanation_sft*`,
`data/processed/sft_{train,val}.jsonl`, `data/processed/preference_pairs.jsonl`,
`checkpoints/mlx_lora_verifier_300/` — all feed into `run_final_evaluation_suite.py` →
`final_results.json` (`explanation`, `preference_reranking` sections), which README cites.

One naming wrinkle: `scripts/train_mlx_lora.py` (Makefile `train-mlx-lora`/`eval-mlx-lora`) and
`scripts/train_mlx_lora_explanation.py` (no Makefile target, only reachable via its own config)
both exist and aren't obviously the same thing. Worth a quick look later, but not blocking —
both produce checkpoints that are referenced in `final_results.json`.

**Recommendation:** keep, no action needed now.

---

## G. `data/processed/` — category 1 (all in active use)

Checked every file: small (`_train/_val/_test.jsonl`, `evidence_corpus.jsonl`) and `_large`
variants are both used (small for the sample pipeline + tests, large for Phases 2–6). Retriever/
reranker/verifier pair files, `explanation_sft*`, `sft_*`, `preference_pairs.jsonl`,
`mlx_lora/*` — all have a producing script and a consuming script/report. No orphaned data files
found.

**Recommendation:** no action.

---

## H. `reports/` — category 1

README explicitly frames all of `reports/` as "checked-in sample-scale evaluation runs" that
back the numbers in the README/PROJECT_SUMMARY. Every report file traced back to either a
Makefile target or `run_final_evaluation_suite.py`. The one exception is Section A
(`retrieved_augmented_verifier_eval.*`, from the buggy run, uncommitted).

**Recommendation:** no action beyond Section A.

---

## I. Low-risk mechanical cleanup — category 2, no decision needed

- All `__pycache__/` directories outside `.venv/` (≈1.5MB total, gitignored, untracked).
- `.pytest_cache/` (28KB, gitignored, untracked).
- Root `__pycache__/{app,cli}.cpython-313.pyc`.

**Recommendation:** delete immediately — purely generated, gitignored, zero risk, doesn't even
touch git history.

---

## J. `notebooks/` — category 1

Empty except `.gitkeep`. Fine as a placeholder; no action.

---

## K. Summary of decisions needed from you

1. **Phase 6 (Section A):** keep the in-progress fix and discard only the stale buggy
   report/checkpoint, or discard the whole Phase 6 branch of work (including the
   `train_transformer_verifier_clean.py` diff)?
2. **Legacy "non-clean" verifier generation (Sections B & C):** retire
   `train_verifier.py`/`checkpoints/verifier` and
   `train_transformer_verifier.py`/`checkpoints/transformer_verifier` (with small follow-on
   edits to `core/config.py`, `Makefile`, and one test import), or keep them as historical
   baselines?
3. **`_final` duplication (Section C):** retire `eval_faithfulness.py`/`pareto_analysis.py`
   (non-final) and repoint `make eval-faithfulness`/`make pareto-analysis` at the `_final`
   versions, or keep both?
4. **`docs/archive/` (Section E):** keep (already explicitly scoped as historical archive) or
   remove for a fully clean slate?
5. **Dead scaffolding (Section D):** remove `training/train_*.py` stubs,
   `models/roberta_baseline.py`, `data/build_preference_pairs.py` (with a small
   `tests/test_imports.py` edit)?

Section I (`__pycache__`/`.pytest_cache`) needs no decision — happy to clean that immediately
once you give the go-ahead on the rest, or as part of the first small commit.

Nothing has been deleted or modified yet (aside from the pre-existing uncommitted diff in
`scripts/train_transformer_verifier_clean.py` from before this audit started).
