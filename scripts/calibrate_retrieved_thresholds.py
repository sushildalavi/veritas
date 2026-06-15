"""Calibrate per_passage_max thresholds on retrieved evidence.

Caches the raw per-passage-max SUPPORTED/REFUTED/NOT ENOUGH INFO scores once per
example (retrieved evidence, current serving profile), then grid-searches
support_threshold, refute_threshold, and an optional support-vs-refute margin to
maximize macro-F1 on a calibration split (default: fever_val_large +
scifact_val_large), and evaluates the best combination -- alongside the current
default thresholds -- on a held-out split (default: fever_test_large +
scifact_test_large, the same 650-example set used for the headline reports).

The decision rule (generalizes the current per_passage_max rule, which is the
margin=0 case):

    if max_support >= support_threshold and (max_support - max_refute) >= margin:
        SUPPORTED
    elif max_refute >= refute_threshold and (max_refute - max_support) >= margin:
        REFUTED
    else:
        NOT ENOUGH INFO
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.config import load_project_settings
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import iter_records, load_evidence_corpus, load_records
from models.model_router import ModelRouter
from scripts.eval_oracle_vs_retrieved_v2 import (
    LABEL_ORDER,
    _dataset_source,
    _label_distribution,
    _load_eval_records,
    _normalize_label,
)
from serving.model_loader import _load_reranker_runtime, _load_retrieval_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate retrieved-evidence verifier thresholds.")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--suffix", default="_large")
    parser.add_argument("--calibration-split-prefixes", default="fever_val,scifact_val")
    parser.add_argument("--holdout-split-prefixes", default="fever_test,scifact_test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--support-grid", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--refute-grid", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--margin-grid", default="0.0,0.02,0.05,0.1")
    parser.add_argument("--report-json", default="reports/threshold_calibration_650.json")
    parser.add_argument("--report-md", default="reports/threshold_calibration_650.md")
    return parser


def _collect_scores(
    *,
    settings,
    checkpoint: str,
    data_dir: Path,
    suffix: str,
    split_prefixes: tuple[str, ...],
    top_k: int,
    max_examples: int,
) -> list[dict[str, object]]:
    records = _load_eval_records(data_dir, suffix=suffix, split_prefixes=split_prefixes, max_examples=max_examples)
    corpus = load_evidence_corpus(data_dir, suffix=suffix)
    retrieval_runtime = _load_retrieval_runtime(corpus, settings)
    reranker_runtime = _load_reranker_runtime(settings)
    retrieval_depth = max(top_k, settings.rrf_top_k, settings.rerank_top_k, settings.final_top_k, 50)

    prefer_deberta = settings.verifier_backend.lower() != "mock"
    router = ModelRouter(
        verifier_checkpoint=checkpoint,
        prefer_deberta=prefer_deberta,
        aggregation_mode="per_passage_max",
        support_threshold=settings.support_threshold,
        refute_threshold=settings.refute_threshold,
    )

    cached: list[dict[str, object]] = []
    for record in records:
        hits = retrieval_runtime.retriever.retrieve(record.claim, top_k=retrieval_depth)
        if reranker_runtime.reranker is not None:
            hits = list(reranker_runtime.reranker.rank(record.claim, hits))
        result = router.predict(record.claim, hits[:top_k])
        cached.append(
            {
                "label": _normalize_label(record.label),
                "dataset": str(record.metadata.get("source", "unknown")),
                "max_support": float(result.logits.get("SUPPORTED", 0.0)),
                "max_refute": float(result.logits.get("REFUTED", 0.0)),
                "max_nei": float(result.logits.get("NOT ENOUGH INFO", 0.0)),
            }
        )
    return cached


def _predict_label(scores: dict[str, object], support_threshold: float, refute_threshold: float, margin: float) -> str:
    support = scores["max_support"]
    refute = scores["max_refute"]
    if support >= support_threshold and (support - refute) >= margin:
        return "SUPPORTED"
    if refute >= refute_threshold and (refute - support) >= margin:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def _evaluate_thresholds(
    cached: list[dict[str, object]],
    support_threshold: float,
    refute_threshold: float,
    margin: float,
) -> dict[str, object]:
    labels = [c["label"] for c in cached]
    predictions = [_predict_label(c, support_threshold, refute_threshold, margin) for c in cached]
    report = classification_report(labels, predictions, labels=LABEL_ORDER, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=LABEL_ORDER).tolist()
    refuted_predicted = sum(1 for p in predictions if p == "REFUTED")
    return {
        "support_threshold": support_threshold,
        "refute_threshold": refute_threshold,
        "margin": margin,
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, labels=LABEL_ORDER, average="macro", zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
            }
            for label in LABEL_ORDER
        },
        "confusion_matrix": matrix,
        "refuted_prediction_rate": round(refuted_predicted / len(predictions), 4) if predictions else 0.0,
    }


def _grid_search(cached: list[dict[str, object]], support_grid: list[float], refute_grid: list[float], margin_grid: list[float]) -> dict[str, object]:
    best = None
    for margin in margin_grid:
        for support_threshold in support_grid:
            for refute_threshold in refute_grid:
                result = _evaluate_thresholds(cached, support_threshold, refute_threshold, margin)
                if best is None or result["macro_f1"] > best["macro_f1"]:
                    best = result
    return best


def _git_commit_hash() -> str | None:
    try:
        return (
            subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
            .stdout.strip()
            or None
        )
    except Exception:
        return None


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Retrieved-Evidence Threshold Calibration",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Git commit: `{report.get('git_commit')}`",
        f"- Config path: `{report['config_path']}`",
        f"- Calibration split: `{report['calibration_source']}` ({report['calibration_size']} examples)",
        f"- Held-out split: `{report['holdout_source']}` ({report['holdout_size']} examples)",
        f"- Retrieval backend: `{report['retrieval_backend']}`",
        "",
        "## Calibration grid search",
        "",
        f"- Grid: support={report['grids']['support_grid']}, refute={report['grids']['refute_grid']}, margin={report['grids']['margin_grid']}",
        f"- Default thresholds (support={report['default']['support_threshold']}, refute={report['default']['refute_threshold']}, margin=0.0): "
        f"calibration macro_f1={report['default_on_calibration']['macro_f1']}",
        f"- Best on calibration: support={report['best']['support_threshold']}, refute={report['best']['refute_threshold']}, "
        f"margin={report['best']['margin']}, macro_f1={report['best']['macro_f1']}",
        "",
        "## Held-out (650-example test set) comparison",
        "",
        "| setting | support | refute | margin | accuracy | macro_f1 | SUPPORTED F1 | REFUTED F1 | NEI F1 | refuted_pred_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("default_on_holdout", "best_on_holdout"):
        m = report[name]
        pc = m["per_class"]
        lines.append(
            f"| {name} | {m['support_threshold']} | {m['refute_threshold']} | {m['margin']} | {m['accuracy']} | {m['macro_f1']} | "
            f"{pc['SUPPORTED']['f1']} | {pc['REFUTED']['f1']} | {pc['NOT ENOUGH INFO']['f1']} | {m['refuted_prediction_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Delta (best vs. default, held-out)",
            "",
            f"- macro_f1 delta: {report['holdout_delta']['macro_f1']}",
            f"- accuracy delta: {report['holdout_delta']['accuracy']}",
            f"- refuted_prediction_rate delta: {report['holdout_delta']['refuted_prediction_rate']}",
            "",
            f"**Verdict: {report['verdict']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings(args.config)
    checkpoint = args.checkpoint or settings.transformer_clean_checkpoint
    data_dir = Path(args.data_dir)

    calibration_prefixes = tuple(p.strip() for p in args.calibration_split_prefixes.split(",") if p.strip())
    holdout_prefixes = tuple(p.strip() for p in args.holdout_split_prefixes.split(",") if p.strip())

    support_grid = [round(float(x), 4) for x in args.support_grid.split(",")]
    refute_grid = [round(float(x), 4) for x in args.refute_grid.split(",")]
    margin_grid = [round(float(x), 4) for x in args.margin_grid.split(",")]

    started = perf_counter()
    calibration_cached = _collect_scores(
        settings=settings,
        checkpoint=checkpoint,
        data_dir=data_dir,
        suffix=args.suffix,
        split_prefixes=calibration_prefixes,
        top_k=args.top_k,
        max_examples=args.max_examples,
    )
    holdout_cached = _collect_scores(
        settings=settings,
        checkpoint=checkpoint,
        data_dir=data_dir,
        suffix=args.suffix,
        split_prefixes=holdout_prefixes,
        top_k=args.top_k,
        max_examples=args.max_examples,
    )
    runtime_seconds = round(perf_counter() - started, 3)

    default_support = settings.support_threshold
    default_refute = settings.refute_threshold

    default_on_calibration = _evaluate_thresholds(calibration_cached, default_support, default_refute, 0.0)
    best = _grid_search(calibration_cached, support_grid, refute_grid, margin_grid)

    default_on_holdout = _evaluate_thresholds(holdout_cached, default_support, default_refute, 0.0)
    best_on_holdout = _evaluate_thresholds(holdout_cached, best["support_threshold"], best["refute_threshold"], best["margin"])

    macro_f1_delta = round(best_on_holdout["macro_f1"] - default_on_holdout["macro_f1"], 4)
    if macro_f1_delta > 0:
        verdict = f"held-out macro_f1 improved by {macro_f1_delta} ({default_on_holdout['macro_f1']} -> {best_on_holdout['macro_f1']})"
    elif macro_f1_delta == 0:
        verdict = "no change in held-out macro_f1; current thresholds remain best"
    else:
        verdict = (
            f"held-out macro_f1 did NOT improve (delta {macro_f1_delta}); the calibration-best thresholds overfit the "
            "calibration split and should not replace the current defaults"
        )

    report = {
        "headline": f"threshold_calibration complete: {verdict}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "checkpoint": checkpoint,
        "config_path": args.config,
        "retrieval_backend": settings.retrieval_backend,
        "calibration_source": _dataset_source(calibration_prefixes, args.suffix),
        "holdout_source": _dataset_source(holdout_prefixes, args.suffix),
        "calibration_size": len(calibration_cached),
        "holdout_size": len(holdout_cached),
        "calibration_label_distribution": _label_distribution([c["label"] for c in calibration_cached]),
        "holdout_label_distribution": _label_distribution([c["label"] for c in holdout_cached]),
        "runtime_seconds": runtime_seconds,
        "grids": {
            "support_grid": support_grid,
            "refute_grid": refute_grid,
            "margin_grid": margin_grid,
        },
        "default": {"support_threshold": default_support, "refute_threshold": default_refute},
        "default_on_calibration": default_on_calibration,
        "best": best,
        "default_on_holdout": default_on_holdout,
        "best_on_holdout": best_on_holdout,
        "holdout_delta": {
            "accuracy": round(best_on_holdout["accuracy"] - default_on_holdout["accuracy"], 4),
            "macro_f1": macro_f1_delta,
            "refuted_prediction_rate": round(best_on_holdout["refuted_prediction_rate"] - default_on_holdout["refuted_prediction_rate"], 4),
        },
        "verdict": verdict,
    }

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(report["headline"])


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
