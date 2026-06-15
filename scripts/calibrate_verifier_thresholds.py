"""Calibrate retrieved-evidence verifier thresholds on held-out data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, classification_report, f1_score

from core.config import ProjectSettings, load_project_settings
from data.schemas import ClaimEvidenceRecord
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import iter_records, load_evidence_corpus, load_records
from models.model_router import ModelRouter
from serving.model_loader import _load_retrieval_runtime, _load_reranker_runtime

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
DEFAULT_SPLIT_PREFIXES = ("fever_val", "scifact_val")


@dataclass(frozen=True)
class PassageScoreSummary:
    support: float
    refute: float
    nei: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate verifier thresholds on retrieved evidence.")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--suffix", default="_large")
    parser.add_argument("--split-prefixes", default="fever_val,scifact_val")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--threshold-config-out", default="configs/verifier_thresholds.json")
    parser.add_argument("--report-json", default="reports/verifier_threshold_calibration.json")
    parser.add_argument("--report-md", default="reports/verifier_threshold_calibration.md")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings(args.config)
    checkpoint = args.checkpoint or settings.transformer_clean_checkpoint
    split_prefixes = _parse_prefixes(args.split_prefixes)
    report = calibrate_verifier_thresholds(
        settings=settings,
        checkpoint=checkpoint,
        data_dir=Path(args.data_dir),
        suffix=args.suffix,
        split_prefixes=split_prefixes,
        top_k=args.top_k,
        max_examples=args.max_examples,
        config_path=args.config,
    )
    threshold_path = Path(args.threshold_config_out)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "config_path": args.config,
                "git_commit": report["git_commit"],
                "status": "diagnostic_calibrated_from_validation_slice",
                "calibration_split_prefixes": report["split_prefixes"],
                "support_threshold": report["best"]["support_threshold"],
                "refute_threshold": report["best"]["refute_threshold"],
                "timestamp_utc": report["timestamp_utc"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(report["headline"])


def calibrate_verifier_thresholds(
    *,
    settings: ProjectSettings,
    checkpoint: str,
    data_dir: Path,
    suffix: str,
    split_prefixes: tuple[str, ...] = DEFAULT_SPLIT_PREFIXES,
    top_k: int,
    max_examples: int = 0,
    config_path: str | None = None,
) -> dict[str, object]:
    records = _load_eval_records(data_dir, suffix=suffix, split_prefixes=split_prefixes, max_examples=max_examples)
    corpus = load_evidence_corpus(data_dir, suffix=suffix)
    retrieval_runtime = _load_retrieval_runtime(corpus, settings)
    reranker_runtime = _load_reranker_runtime(settings)
    prefer_deberta = settings.verifier_backend.lower() != "mock"
    router = ModelRouter(
        verifier_checkpoint=checkpoint,
        prefer_deberta=prefer_deberta,
        aggregation_mode="per_passage_max",
        support_threshold=settings.support_threshold,
        refute_threshold=settings.refute_threshold,
    )

    labels = [_normalize_label(record.label) for record in records]
    retrieval_depth = max(top_k, settings.rrf_top_k, settings.rerank_top_k, settings.final_top_k, 50)
    score_summaries: list[PassageScoreSummary] = []
    for record in records:
        hits = retrieval_runtime.retriever.retrieve(record.claim, top_k=retrieval_depth)
        if reranker_runtime.reranker is not None:
            hits = list(reranker_runtime.reranker.rank(record.claim, hits))
        passage_scores = router.score_evidence_passages(record.claim, hits[:top_k])
        score_summaries.append(_summarize_scores(passage_scores))

    baseline_predictions = [
        _aggregate_label(summary, settings.support_threshold, settings.refute_threshold) for summary in score_summaries
    ]
    baseline_metrics = _classification_metrics(labels, baseline_predictions)

    grid = [round(value, 2) for value in _float_range(0.3, 0.8, 0.05)]
    candidates = []
    for support_threshold in grid:
        for refute_threshold in grid:
            predictions = [_aggregate_label(summary, support_threshold, refute_threshold) for summary in score_summaries]
            metrics = _classification_metrics(labels, predictions)
            candidates.append(
                {
                    "support_threshold": support_threshold,
                    "refute_threshold": refute_threshold,
                    **metrics,
                }
            )
    candidates.sort(key=lambda item: (item["macro_f1"], item["accuracy"]), reverse=True)
    best = candidates[0]

    return {
        "headline": (
            f"threshold calibration complete: macro_f1 {baseline_metrics['macro_f1']:.3f} -> "
            f"{best['macro_f1']:.3f} at support={best['support_threshold']:.2f}, refute={best['refute_threshold']:.2f}"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "checkpoint": checkpoint,
        "config_path": config_path,
        "dataset_source": _dataset_source(split_prefixes, suffix),
        "retrieval_backend": retrieval_runtime.retrieval_backend,
        "reranker_backend": reranker_runtime.reranker_backend,
        "sample_size": len(records),
        "split_prefixes": list(split_prefixes),
        "suffix": suffix,
        "top_k": top_k,
        "baseline": {
            "support_threshold": settings.support_threshold,
            "refute_threshold": settings.refute_threshold,
            **baseline_metrics,
        },
        "best": best,
        "top_candidates": candidates[:10],
    }


def _load_eval_records(
    data_dir: Path,
    *,
    suffix: str,
    split_prefixes: tuple[str, ...],
    max_examples: int,
) -> list[ClaimEvidenceRecord]:
    split_names = [f"{prefix}{suffix}" for prefix in split_prefixes]
    records_by_split = load_records(data_dir, split_names=split_names)
    records = [record for record in iter_records(records_by_split, split_prefixes) if record.claim.strip()]
    if max_examples > 0:
        return records[:max_examples]
    return records


def _summarize_scores(passage_scores) -> PassageScoreSummary:  # noqa: ANN001
    return PassageScoreSummary(
        support=max((score.logits.get("SUPPORTED", 0.0) for score in passage_scores), default=0.0),
        refute=max((score.logits.get("REFUTED", 0.0) for score in passage_scores), default=0.0),
        nei=max((score.logits.get("NOT ENOUGH INFO", 0.0) for score in passage_scores), default=1.0),
    )


def _aggregate_label(summary: PassageScoreSummary, support_threshold: float, refute_threshold: float) -> str:
    if summary.support >= support_threshold and summary.support > summary.refute:
        return "SUPPORTED"
    if summary.refute >= refute_threshold and summary.refute > summary.support:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def _classification_metrics(labels: list[str], predictions: list[str]) -> dict[str, object]:
    report = classification_report(labels, predictions, labels=LABEL_ORDER, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, labels=LABEL_ORDER, average="macro", zero_division=0)), 4),
        "nei_false_positive_rate": round(_nei_false_positive_rate(labels, predictions), 4),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
            }
            for label in LABEL_ORDER
        },
    }


def _nei_false_positive_rate(labels: list[str], predictions: list[str]) -> float:
    nei_total = sum(1 for label in labels if label == "NOT ENOUGH INFO")
    if nei_total == 0:
        return 0.0
    nei_false_positives = sum(1 for truth, pred in zip(labels, predictions) if truth == "NOT ENOUGH INFO" and pred != "NOT ENOUGH INFO")
    return nei_false_positives / nei_total


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def _parse_prefixes(raw: str) -> tuple[str, ...]:
    prefixes = tuple(part.strip() for part in raw.split(",") if part.strip())
    return prefixes or DEFAULT_SPLIT_PREFIXES


def _dataset_source(split_prefixes: tuple[str, ...], suffix: str) -> str:
    joined = ", ".join(f"{prefix}{suffix}" for prefix in split_prefixes)
    return f"structured records from {joined}"


def _git_commit_hash() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            or None
        )
    except Exception:
        return None


def _float_range(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def _to_markdown(report: dict[str, object]) -> str:
    baseline = report["baseline"]
    best = report["best"]
    lines = [
        "# Verifier Threshold Calibration",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Git commit: `{report.get('git_commit')}`",
        f"- Config path: `{report.get('config_path')}`",
        f"- Dataset source: `{report.get('dataset_source')}`",
        f"- Retrieval backend: `{report['retrieval_backend']}`",
        f"- Reranker backend: `{report['reranker_backend']}`",
        f"- Split prefixes: `{', '.join(report['split_prefixes'])}`",
        f"- Sample size: {report['sample_size']}",
        f"- Top-k retrieved evidence: {report['top_k']}",
        "",
        "## Summary",
        "",
        f"- Baseline thresholds: support={baseline['support_threshold']}, refute={baseline['refute_threshold']}",
        f"- Baseline macro_f1: {baseline['macro_f1']}",
        f"- Best thresholds: support={best['support_threshold']}, refute={best['refute_threshold']}",
        f"- Best macro_f1: {best['macro_f1']}",
        "",
        "## Top candidates",
        "",
        "| support | refute | accuracy | macro_f1 | nei_fp_rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for candidate in report["top_candidates"]:
        lines.append(
            f"| {candidate['support_threshold']} | {candidate['refute_threshold']} | "
            f"{candidate['accuracy']} | {candidate['macro_f1']} | {candidate['nei_false_positive_rate']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
