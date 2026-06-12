"""Final Pareto analysis: verifier quality vs. deployment cost.

Compares the verifiers that actually exist as trained checkpoints
(sklearn TF-IDF/LogReg baseline and the clean class-weighted DistilRoBERTa
transformer) using their real test-set metrics from
``reports/verifier_clean_baseline.json`` and
``reports/transformer_verifier_clean_eval.json``.

QLoRA and DPO verifiers are not included with measured numbers because their
training is blocked on this machine (no CUDA GPU / bitsandbytes - see
``reports/qlora_BLOCKED_GPU_REQUIRED.md`` and
``reports/dpo_BLOCKED_QLORA_REQUIRED.md``). Their rows are recorded as
``status: "not trained - GPU required"`` rather than fabricated.

Outputs:
  reports/final_pareto_analysis.{json,md}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from evaluation.pareto_analysis import ParetoPoint, pareto_frontier, pareto_report
from evaluation.reporting import write_report

_SKLEARN_BENCHMARK_TEXT = "Claim: Example claim text here.\nEvidence: Example evidence text here for testing latency."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final Pareto analysis across trained verifiers.")
    parser.add_argument("--sklearn-checkpoint", default="checkpoints/verifier_clean")
    parser.add_argument("--sklearn-report", default="reports/verifier_clean_baseline.json")
    parser.add_argument("--transformer-checkpoint", default="checkpoints/transformer_verifier_clean")
    parser.add_argument("--transformer-report", default="reports/transformer_verifier_clean_eval.json")
    parser.add_argument("--report-json", default="reports/final_pareto_analysis.json")
    parser.add_argument("--report-md", default="reports/final_pareto_analysis.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()

    sklearn_report = json.loads(Path(args.sklearn_report).read_text(encoding="utf-8"))
    transformer_report = json.loads(Path(args.transformer_report).read_text(encoding="utf-8"))

    sklearn_memory_mb = _checkpoint_size_mb(Path(args.sklearn_checkpoint))
    transformer_memory_mb = _checkpoint_size_mb(Path(args.transformer_checkpoint))
    sklearn_latency_ms = _sklearn_latency_ms(Path(args.sklearn_checkpoint) / "model.joblib")
    transformer_latency_ms = float(transformer_report["test"]["latency_ms_per_example"])

    configs = [
        ("sklearn-tfidf-logreg", float(sklearn_report["test"]["macro_f1"]), sklearn_latency_ms, sklearn_memory_mb),
        ("distilroberta-clean", float(transformer_report["test"]["macro_f1"]), transformer_latency_ms, transformer_memory_mb),
    ]

    points: list[ParetoPoint] = []
    summaries: list[dict[str, object]] = []
    for model_name, macro_f1, latency_ms, memory_mb in configs:
        feasibility = max(0.0, 1.0 - (latency_ms / 1000.0) - (memory_mb / 500.0))
        points.append(
            ParetoPoint(
                model=model_name,
                macro_f1=macro_f1,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                deployment_feasibility=feasibility,
            )
        )
        summaries.append(
            {
                "model": model_name,
                "macro_f1": macro_f1,
                "latency_ms": latency_ms,
                "memory_mb": memory_mb,
                "deployment_feasibility": feasibility,
                "status": "measured",
            }
        )

    not_trained = [
        {
            "model": "qlora-tinyllama",
            "status": "not trained - GPU required",
            "note": "see reports/qlora_BLOCKED_GPU_REQUIRED.md",
        },
        {
            "model": "dpo-tinyllama",
            "status": "not trained - QLoRA required",
            "note": "see reports/dpo_BLOCKED_QLORA_REQUIRED.md",
        },
    ]

    report = {
        "points": pareto_report(points),
        "frontier": pareto_report(pareto_frontier(points)),
        "summaries": summaries,
        "not_trained": not_trained,
    }
    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(f"frontier: {report['frontier']}")


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


def _sklearn_latency_ms(model_path: Path, repeats: int = 50) -> float:
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    texts = [_SKLEARN_BENCHMARK_TEXT] * repeats
    started = time.perf_counter()
    pipeline.predict(texts)
    return (time.perf_counter() - started) * 1000.0 / repeats


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Final Pareto Analysis",
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
                    f"{point['latency_ms']:.4f}",
                    f"{point['memory_mb']:.2f}",
                    f"{point['deployment_feasibility']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## All measured points", "", "| model | macro_f1 | latency_ms | memory_mb | deployment_feasibility |", "| --- | --- | --- | --- | --- |"])
    for point in report["points"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(point["model"]),
                    f"{point['macro_f1']:.3f}",
                    f"{point['latency_ms']:.4f}",
                    f"{point['memory_mb']:.2f}",
                    f"{point['deployment_feasibility']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not trained (blocked)", "", "| model | status | note |", "| --- | --- | --- |"])
    for entry in report["not_trained"]:
        lines.append(f"| {entry['model']} | {entry['status']} | {entry['note']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
