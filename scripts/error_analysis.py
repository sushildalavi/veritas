"""Generate an error analysis report from verifier predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.error_analysis import ErrorExample, build_error_analysis, categorize_claim, write_error_report
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_records
from serving.model_loader import load_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verifier error analysis on sampled claims.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(evidence_corpus_path=data_dir / "evidence_corpus.jsonl")
    records_by_split = load_records(data_dir)

    examples: list[ErrorExample] = []
    rows: list[dict[str, object]] = []
    for split_name, records in records_by_split.items():
        for record in records:
            outcome = pipeline.reflection_loop.run(record.claim, top_k=args.top_k)
            prediction = outcome.verification.verdict if outcome.verification else "NOT ENOUGH INFO"
            category = categorize_claim(record.claim)
            row = {
                "split": split_name,
                "claim_id": record.claim_id,
                "claim": record.claim,
                "truth": record.label,
                "prediction": prediction,
                "category": category,
                "correct": _normalize_label(prediction) == _normalize_label(record.label),
            }
            rows.append(row)
            if not row["correct"]:
                examples.append(
                    ErrorExample(
                        claim=record.claim,
                        truth=record.label,
                        prediction=prediction,
                        category=category,
                    )
                )

    report = build_error_analysis(examples)
    report["top_k"] = args.top_k
    report["backend"] = pipeline.verifier_backend
    report["mismatch_count"] = len(examples)
    report["total_examples"] = len(rows)
    report["error_rate"] = (len(examples) / len(rows)) if rows else 0.0
    write_report(report, reports_dir / "error_analysis_summary.json")
    write_error_report(examples, reports_dir)
    (reports_dir / "error_analysis.md").write_text(_to_markdown(report, examples), encoding="utf-8")
    print("Wrote error analysis to reports/error_analysis_summary.json")


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def _to_markdown(report: dict[str, object], examples: list[ErrorExample]) -> str:
    lines = [
        "# Error Analysis",
        "",
        f"- Backend: {report['backend']}",
        f"- Top K: {report['top_k']}",
        f"- Error rate: {report['error_rate']:.3f}",
        "",
        "## Category Counts",
        "",
        "| category | count |",
        "| --- | --- |",
    ]
    for category, count in sorted(report.get("category_counts", {}).items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Examples", "", "| claim | truth | prediction | category |", "| --- | --- | --- | --- |"])
    for example in examples[:10]:
        lines.append(f"| {example.claim} | {example.truth} | {example.prediction} | {example.category} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
