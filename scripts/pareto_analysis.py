"""Generate a Pareto report for verifier quality and deployment cost."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

from evaluation.pareto_analysis import ParetoPoint, pareto_frontier, pareto_report
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_records
from sklearn.metrics import f1_score
from serving.model_loader import load_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pareto analysis for verifier configurations.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--top-ks", nargs="*", type=int, default=[3, 5])
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records_by_split = load_records(data_dir)
    records = [record for split_records in records_by_split.values() for record in split_records]

    configs = [
        ("mock", load_pipeline(evidence_corpus_path=data_dir / "evidence_corpus.jsonl", verifier_checkpoint="/tmp/missing-verifier"), 0.0),
        ("trained", load_pipeline(evidence_corpus_path=data_dir / "evidence_corpus.jsonl"), _checkpoint_size_mb(Path("checkpoints/verifier"))),
    ]

    points: list[ParetoPoint] = []
    summaries: list[dict[str, object]] = []
    for config_name, pipeline, memory_mb in configs:
        for top_k in args.top_ks:
            metrics = _evaluate_pipeline(pipeline, records, top_k)
            latency_ms = metrics["mean_latency_ms"]
            quality = metrics["macro_f1"]
            feasibility = max(0.0, 1.0 - (latency_ms / 1000.0) - (memory_mb / 500.0))
            model_name = f"{config_name}-top{top_k}"
            point = ParetoPoint(
                model=model_name,
                macro_f1=quality,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                deployment_feasibility=feasibility,
            )
            points.append(point)
            summaries.append({"model": model_name, **metrics, "memory_mb": memory_mb, "deployment_feasibility": feasibility})

    report = {
        "points": pareto_report(points),
        "frontier": pareto_report(pareto_frontier(points)),
        "summaries": summaries,
    }
    write_report(report, reports_dir / "pareto_analysis.json")
    (reports_dir / "pareto_analysis.md").write_text(_to_markdown(report), encoding="utf-8")
    print("Wrote Pareto analysis to reports/pareto_analysis.json")


def _evaluate_pipeline(pipeline, records, top_k: int) -> dict[str, float]:
    predictions: list[str] = []
    truths: list[str] = []
    latencies: list[float] = []
    for record in records:
        start = time.perf_counter()
        outcome = pipeline.reflection_loop.run(record.claim, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions.append(outcome.verification.verdict if outcome.verification else "NOT ENOUGH INFO")
        truths.append(_normalize_label(record.label))
    return {
        "macro_f1": float(f1_score(truths, predictions, average="macro")),
        "mean_latency_ms": float(mean(latencies)) if latencies else 0.0,
        "example_count": float(len(records)),
    }


def _checkpoint_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024 * 1024)


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Pareto Analysis",
        "",
        "## Frontier",
        "",
        "| model | macro_f1 | latency_ms | memory_mb | deployment_feasibility |",
        "| --- | --- | --- | --- | --- |",
    ]
    for point in report["frontier"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(point["model"]),
                    f"{point['macro_f1']:.3f}",
                    f"{point['latency_ms']:.2f}",
                    f"{point['memory_mb']:.2f}",
                    f"{point['deployment_feasibility']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## All Points", "", "| model | macro_f1 | latency_ms | memory_mb | deployment_feasibility |", "| --- | --- | --- | --- | --- |"])
    for point in report["points"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(point["model"]),
                    f"{point['macro_f1']:.3f}",
                    f"{point['latency_ms']:.2f}",
                    f"{point['memory_mb']:.2f}",
                    f"{point['deployment_feasibility']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
