"""Evaluate citation faithfulness on the sampled verification set."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_records
from rag import build_context, check_citations, generate_template_explanation
from serving.model_loader import load_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run faithfulness evaluation on sampled claims.")
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

    rows: list[dict[str, object]] = []
    for split_name, records in records_by_split.items():
        for record in records:
            outcome = pipeline.reflection_loop.run(record.claim, top_k=args.top_k)
            context = build_context(record.claim, outcome.evidence, top_k=args.top_k)
            verification = outcome.verification or pipeline.verifier.predict(record.claim, outcome.evidence)
            explanation_output = generate_template_explanation(context, verification)
            citation_result = check_citations(explanation_output.explanation, context)
            rows.append(
                {
                    "split": split_name,
                    "claim_id": record.claim_id,
                    "truth": record.label,
                    "prediction": verification.verdict,
                    "decision": outcome.decision,
                    "citation_valid": citation_result.valid,
                    "citation_precision": citation_result.citation_precision,
                    "unsupported_sentence_rate": citation_result.unsupported_sentence_rate,
                    "verdict_consistency": citation_result.verdict_consistency,
                    "retries_used": outcome.retries_used,
                }
            )

    report = {
        "top_k": args.top_k,
        "backend": pipeline.verifier_backend,
        "overall": _summarize(rows),
        "splits": {},
        "examples": rows,
    }
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split"])].append(row)
    for split_name, split_rows in grouped.items():
        report["splits"][split_name] = _summarize(split_rows)

    write_report(report, reports_dir / "faithfulness_eval.json")
    (reports_dir / "faithfulness_eval.md").write_text(_to_markdown(report), encoding="utf-8")
    print("Wrote faithfulness evaluation to reports/faithfulness_eval.json")


def _summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "rows": 0.0,
            "citation_valid_rate": 0.0,
            "mean_citation_precision": 0.0,
            "mean_unsupported_sentence_rate": 0.0,
            "mean_retries_used": 0.0,
            "verdict_consistency_rate": 0.0,
        }
    return {
        "rows": float(len(rows)),
        "citation_valid_rate": mean(1.0 if row["citation_valid"] else 0.0 for row in rows),
        "mean_citation_precision": mean(float(row["citation_precision"]) for row in rows),
        "mean_unsupported_sentence_rate": mean(float(row["unsupported_sentence_rate"]) for row in rows),
        "mean_retries_used": mean(float(row["retries_used"]) for row in rows),
        "verdict_consistency_rate": mean(1.0 if row["verdict_consistency"] else 0.0 for row in rows),
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Faithfulness Evaluation",
        "",
        f"- Backend: {report['backend']}",
        f"- Top K: {report['top_k']}",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for metric_name, value in report["overall"].items():
        lines.append(f"| {metric_name} | {value:.3f} |")

    for split_name, metrics in report["splits"].items():
        lines.extend(["", f"## {split_name}", "", "| metric | value |", "| --- | --- |"])
        for metric_name, value in metrics.items():
            lines.append(f"| {metric_name} | {value:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
