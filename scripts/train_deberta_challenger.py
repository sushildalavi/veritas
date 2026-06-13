"""Evaluate a DeBERTa challenger verifier against the DistilRoBERTa baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from models.deberta_verifier import DebertaVerifier
from models.labels import normalize_label
from data.schemas import EvidenceSpan
from sklearn.metrics import f1_score, recall_score


LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the DeBERTa verifier challenger.")
    parser.add_argument("--checkpoint", default="checkpoints/deberta_verifier_clean")
    parser.add_argument("--baseline-checkpoint", default="checkpoints/transformer_verifier_clean")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--eval-split", default="verifier_val.jsonl")
    parser.add_argument("--test-split", default="verifier_test.jsonl")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--report-json", default="reports/deberta_verifier_clean_eval.json")
    parser.add_argument("--report-md", default="reports/deberta_verifier_clean_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    checkpoint = Path(args.checkpoint)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        attempted_path = report_md.with_name("deberta_challenger_ATTEMPTED.md")
        attempted_path.write_text(
            "\n".join(
                [
                    "# DeBERTa Challenger Attempted",
                    "",
                    f"- checkpoint: {args.checkpoint}",
                    f"- baseline_checkpoint: {args.baseline_checkpoint}",
                    "",
                    "No DeBERTa challenger checkpoint was available in the repo, so the challenger experiment was not run.",
                ]
            ),
            encoding="utf-8",
        )
        raise SystemExit(f"DeBERTa challenger checkpoint not found at {checkpoint}")

    report = _evaluate(checkpoint, Path(args.baseline_checkpoint), Path(args.data_dir), args.eval_split, args.test_split, args.max_examples)
    write_report(report, report_json)
    report_md.write_text(_to_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _evaluate(
    checkpoint: Path,
    baseline_checkpoint: Path,
    data_dir: Path,
    eval_split: str,
    test_split: str,
    max_examples: int,
) -> dict:
    challenger = DebertaVerifier(checkpoint)
    baseline = DebertaVerifier(baseline_checkpoint)
    eval_rows = read_jsonl(data_dir / eval_split)[:max_examples]
    test_rows = read_jsonl(data_dir / test_split)[:max_examples]
    if not eval_rows:
        raise SystemExit(f"No evaluation rows found at {data_dir / eval_split}")

    challenger_eval = _score(challenger, eval_rows)
    baseline_eval = _score(baseline, eval_rows)
    challenger_test = _score(challenger, test_rows) if test_rows else {}
    baseline_test = _score(baseline, test_rows) if test_rows else {}

    model_size_mb = round(checkpoint.stat().st_size / (1024 * 1024), 2) if checkpoint.is_file() else _dir_size_mb(checkpoint)
    return {
        "checkpoint": str(checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "model_size_mb": model_size_mb,
        "eval": {
            "challenger": challenger_eval,
            "baseline": baseline_eval,
        },
        "test": {
            "challenger": challenger_test,
            "baseline": baseline_test,
        },
    }


def _score(verifier: DebertaVerifier, rows: list[dict]) -> dict:
    golds: list[str] = []
    preds: list[str] = []
    latencies: list[float] = []
    for row in rows:
        evidence = [EvidenceSpan(doc_id="E1", text=str(row.get("evidence", "")))]
        started = perf_counter()
        result = verifier.predict(str(row.get("claim", "")), evidence)
        latencies.append(perf_counter() - started)
        golds.append(normalize_label(str(row.get("label", "NOT_ENOUGH_INFO"))))
        preds.append(normalize_label(result.verdict))
    macro_f1 = f1_score(golds, preds, labels=LABEL_ORDER, average="macro", zero_division=0)
    per_class = f1_score(golds, preds, labels=LABEL_ORDER, average=None, zero_division=0)
    refuted_recall = recall_score(golds, preds, labels=LABEL_ORDER, average=None, zero_division=0)[1]
    accuracy = sum(1 for g, p in zip(golds, preds) if g == p) / len(rows)
    return {
        "sample_size": len(rows),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_f1": dict(zip(LABEL_ORDER, (round(float(score), 4) for score in per_class))),
        "refuted_recall": round(float(refuted_recall), 4),
        "mean_latency_seconds": round(mean(latencies), 4),
        "backend": verifier._backend,
        "model_name": verifier.model_name,
    }


def _dir_size_mb(path: Path) -> float:
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return round(total / (1024 * 1024), 2)


def _to_markdown(report: dict) -> str:
    eval_report = report["eval"]
    test_report = report["test"]
    lines = [
        "# DeBERTa Challenger Evaluation",
        "",
        f"- checkpoint: {report['checkpoint']}",
        f"- baseline_checkpoint: {report['baseline_checkpoint']}",
        f"- model_size_mb: {report['model_size_mb']}",
        "",
        "| split | model | accuracy | macro_f1 | refuted_recall | mean_latency_seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| eval | challenger | {eval_report['challenger']['accuracy']} | {eval_report['challenger']['macro_f1']} | {eval_report['challenger']['refuted_recall']} | {eval_report['challenger']['mean_latency_seconds']} |",
        f"| eval | baseline | {eval_report['baseline']['accuracy']} | {eval_report['baseline']['macro_f1']} | {eval_report['baseline']['refuted_recall']} | {eval_report['baseline']['mean_latency_seconds']} |",
    ]
    if test_report:
        lines += [
            f"| test | challenger | {test_report['challenger']['accuracy']} | {test_report['challenger']['macro_f1']} | {test_report['challenger']['refuted_recall']} | {test_report['challenger']['mean_latency_seconds']} |",
            f"| test | baseline | {test_report['baseline']['accuracy']} | {test_report['baseline']['macro_f1']} | {test_report['baseline']['refuted_recall']} | {test_report['baseline']['mean_latency_seconds']} |",
        ]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
