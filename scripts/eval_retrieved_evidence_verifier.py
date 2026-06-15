"""Evaluate the verifier on individual (claim, passage) pairs, with vs without a
lexical-overlap relevance gate.

This is a *new*, finer-grained benchmark built from
``data/processed/retrieved_evidence_verifier_{calibration,holdout}.jsonl``
(see ``scripts/build_retrieved_evidence_dataset.py``). Each example is a single
claim + single retrieved-or-oracle passage, labeled SUPPORTS/REFUTES/NEI based
on whether that passage alone is the claim's gold evidence.

Two predictors are compared on the holdout split (the 650-claim
fever_test_large + scifact_test_large set):

- ``current``: the verifier's per-passage prediction (per_passage_max,
  support_threshold/refute_threshold from `configs/serving.yaml`), unchanged.
- ``gated``: the same verifier prediction, but forced to NEI whenever the
  claim/passage lexical-overlap score (stored per-pair, computable without gold
  labels) is below a threshold tuned on the calibration split
  (fever_val_large + scifact_val_large) to maximize macro-F1.

IMPORTANT: this is a diagnostic benchmark over individual (claim, passage)
pairs, separate from the official 650-claim oracle-vs-retrieved evaluation
(`scripts/eval_oracle_vs_retrieved_v2.py`). A macro-F1 gain here does NOT by
itself constitute a full-650-set improvement -- the gate would need to be
integrated into `ModelRouter` and re-validated on
`reports/oracle_vs_retrieved_v2_full.json` before any such claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.config import load_project_settings
from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from models.model_router import ModelRouter
from scripts.eval_oracle_vs_retrieved_v2 import _git_commit_hash, _normalize_label

PAIR_LABEL_ORDER = ["SUPPORTS", "REFUTES", "NEI"]
_VERDICT_TO_PAIR_LABEL = {"SUPPORTED": "SUPPORTS", "REFUTED": "REFUTES", "NOT ENOUGH INFO": "NEI"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate current vs relevance-gated per-passage verifier.")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--gate-grid", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--report-json", default="reports/retrieved_evidence_verifier_eval.json")
    parser.add_argument("--report-md", default="reports/retrieved_evidence_verifier_eval.md")
    return parser


def _verdict_to_pair_label(verdict: str) -> str:
    return _VERDICT_TO_PAIR_LABEL[_normalize_label(verdict)]


def _predict_current(rows: list[dict[str, object]], router: ModelRouter) -> list[str]:
    predictions = []
    for row in rows:
        span = EvidenceSpan(
            doc_id=str(row["passage_doc_id"]),
            text=str(row["passage_text"]),
            title=row.get("passage_title"),
            score=row.get("retrieval_score"),
        )
        result = router.predict(str(row["claim"]), [span])
        predictions.append(_verdict_to_pair_label(result.verdict))
    return predictions


def _apply_gate(predictions: list[str], rows: list[dict[str, object]], threshold: float) -> list[str]:
    return [
        "NEI" if float(row["lexical_overlap"]) < threshold else pred
        for pred, row in zip(predictions, rows)
    ]


def _evaluate(labels: list[str], predictions: list[str]) -> dict[str, object]:
    report = classification_report(labels, predictions, labels=PAIR_LABEL_ORDER, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=PAIR_LABEL_ORDER).tolist()
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, labels=PAIR_LABEL_ORDER, average="macro", zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
            }
            for label in PAIR_LABEL_ORDER
        },
        "confusion_matrix": matrix,
        "support": len(labels),
    }


def _evaluate_by_source(rows: list[dict[str, object]], predictions: list[str]) -> dict[str, object]:
    labels = [str(row["pair_label"]) for row in rows]
    result = {"overall": _evaluate(labels, predictions)}
    for source in ("fever", "scifact"):
        indices = [i for i, row in enumerate(rows) if row["source"] == source]
        if not indices:
            continue
        result[source] = _evaluate([labels[i] for i in indices], [predictions[i] for i in indices])
    return result


def _grid_search_gate(rows: list[dict[str, object]], current_pred: list[str], grid: list[float]) -> tuple[float, float]:
    labels = [str(row["pair_label"]) for row in rows]
    best_threshold, best_macro_f1 = 0.0, -1.0
    for threshold in grid:
        gated = _apply_gate(current_pred, rows, threshold)
        macro_f1 = f1_score(labels, gated, labels=PAIR_LABEL_ORDER, average="macro", zero_division=0)
        if macro_f1 > best_macro_f1:
            best_threshold, best_macro_f1 = threshold, float(macro_f1)
    return best_threshold, round(best_macro_f1, 4)


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Retrieved-Evidence Per-Passage Verifier Eval (current vs relevance-gated)",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Git commit: `{report.get('git_commit')}`",
        f"- Calibration pairs: {report['calibration_size']}, holdout pairs: {report['holdout_size']}",
        f"- Gate grid: {report['gate_grid']}",
        f"- Best gate threshold (tuned on calibration): {report['best_gate_threshold']} "
        f"(calibration macro_f1: current={report['current_on_calibration']['overall']['macro_f1']}, "
        f"gated={report['best_gate_calibration_macro_f1']})",
        "",
        "## Holdout (650-claim test set, per-passage pairs)",
        "",
        "| predictor | split | accuracy | macro_f1 | SUPPORTS F1 | REFUTES F1 | NEI F1 | support |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for predictor in ("current_on_holdout", "gated_on_holdout"):
        for split_name in ("overall", "fever", "scifact"):
            if split_name not in report[predictor]:
                continue
            m = report[predictor][split_name]
            pc = m["per_class"]
            lines.append(
                f"| {predictor} | {split_name} | {m['accuracy']} | {m['macro_f1']} | {pc['SUPPORTS']['f1']} | "
                f"{pc['REFUTES']['f1']} | {pc['NEI']['f1']} | {m['support']} |"
            )
    lines.extend(
        [
            "",
            "## Delta (gated vs current, holdout overall)",
            "",
            f"- macro_f1 delta: {report['holdout_delta']['macro_f1']}",
            f"- accuracy delta: {report['holdout_delta']['accuracy']}",
            "",
            f"**Verdict: {report['verdict']}**",
            "",
            "_Note: this is a per-passage diagnostic benchmark, separate from the official "
            "650-claim oracle-vs-retrieved evaluation. A gain here does not by itself constitute "
            "a full-650-set improvement._",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings(args.config)
    checkpoint = args.checkpoint or settings.transformer_clean_checkpoint
    data_dir = Path(args.data_dir)
    gate_grid = [round(float(x), 4) for x in args.gate_grid.split(",")]

    calibration_rows = read_jsonl(data_dir / "retrieved_evidence_verifier_calibration.jsonl")
    holdout_rows = read_jsonl(data_dir / "retrieved_evidence_verifier_holdout.jsonl")
    if args.max_pairs:
        calibration_rows = calibration_rows[: args.max_pairs]
        holdout_rows = holdout_rows[: args.max_pairs]

    prefer_deberta = settings.verifier_backend.lower() != "mock"
    router = ModelRouter(
        verifier_checkpoint=checkpoint,
        prefer_deberta=prefer_deberta,
        aggregation_mode="per_passage_max",
        support_threshold=settings.support_threshold,
        refute_threshold=settings.refute_threshold,
    )

    started = perf_counter()
    calibration_pred = _predict_current(calibration_rows, router)
    holdout_pred = _predict_current(holdout_rows, router)
    runtime_seconds = round(perf_counter() - started, 3)

    best_threshold, best_gate_calibration_macro_f1 = _grid_search_gate(calibration_rows, calibration_pred, gate_grid)
    holdout_gated_pred = _apply_gate(holdout_pred, holdout_rows, best_threshold)

    current_on_holdout = _evaluate_by_source(holdout_rows, holdout_pred)
    gated_on_holdout = _evaluate_by_source(holdout_rows, holdout_gated_pred)
    current_on_calibration = _evaluate_by_source(calibration_rows, calibration_pred)

    macro_f1_delta = round(gated_on_holdout["overall"]["macro_f1"] - current_on_holdout["overall"]["macro_f1"], 4)
    if macro_f1_delta > 0:
        verdict = (
            f"per-passage holdout macro_f1 improved by {macro_f1_delta} "
            f"({current_on_holdout['overall']['macro_f1']} -> {gated_on_holdout['overall']['macro_f1']})"
        )
    elif macro_f1_delta == 0:
        verdict = "no change in per-passage holdout macro_f1; relevance gate did not help on this benchmark"
    else:
        verdict = (
            f"per-passage holdout macro_f1 did NOT improve (delta {macro_f1_delta}); "
            "the calibration-tuned gate overfits the calibration split"
        )

    report = {
        "headline": f"retrieved_evidence_verifier_eval complete: {verdict}",
        "git_commit": _git_commit_hash(),
        "checkpoint": checkpoint,
        "config_path": args.config,
        "calibration_size": len(calibration_rows),
        "holdout_size": len(holdout_rows),
        "gate_grid": gate_grid,
        "runtime_seconds": runtime_seconds,
        "best_gate_threshold": best_threshold,
        "best_gate_calibration_macro_f1": best_gate_calibration_macro_f1,
        "current_on_calibration": current_on_calibration,
        "current_on_holdout": current_on_holdout,
        "gated_on_holdout": gated_on_holdout,
        "holdout_delta": {
            "accuracy": round(gated_on_holdout["overall"]["accuracy"] - current_on_holdout["overall"]["accuracy"], 4),
            "macro_f1": macro_f1_delta,
        },
        "verdict": verdict,
    }

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(report["headline"])


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
