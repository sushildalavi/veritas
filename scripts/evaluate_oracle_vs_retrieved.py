"""Evaluate the clean transformer verifier with oracle vs. retrieved evidence.

Oracle evaluation reuses the gold/curated evidence in
``data/processed/verifier_test.jsonl`` (the same evidence the verifier was
trained and evaluated on). End-to-end evaluation replaces that evidence with
the top-1 passage returned by a real BM25 retriever run against
``data/processed/evidence_corpus_large.jsonl`` for the claim text, simulating
what the verifier sees when wired into the retrieval pipeline.

Outputs:
  reports/oracle_verifier_eval.{json,md}
  reports/end_to_end_verifier_eval.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_evidence_corpus, read_jsonl
from retrieval.bm25 import BM25Retriever

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}


def format_input(claim: str, evidence: str) -> str:
    return f"Claim: {claim}\nEvidence: {evidence}"


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate verifier with oracle vs. retrieved evidence.")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_verifier_clean")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--test-file", default="verifier_test.jsonl")
    parser.add_argument("--evidence-corpus-suffix", default="_large")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-errors", type=int, default=10)
    parser.add_argument("--oracle-report-json", default="reports/oracle_verifier_eval.json")
    parser.add_argument("--oracle-report-md", default="reports/oracle_verifier_eval.md")
    parser.add_argument("--end-to-end-report-json", default="reports/end_to_end_verifier_eval.json")
    parser.add_argument("--end-to-end-report-md", default="reports/end_to_end_verifier_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)

    rows = read_jsonl(data_dir / args.test_file)
    if not rows:
        raise SystemExit(f"No test examples found at {data_dir / args.test_file}")

    claims = [str(row.get("claim", "")) for row in rows]
    labels = [_normalize_label(str(row.get("label", "NOT_ENOUGH_INFO"))) for row in rows]
    oracle_evidence = [str(row.get("evidence", "")) for row in rows]

    started = perf_counter()
    corpus = load_evidence_corpus(data_dir, suffix=args.evidence_corpus_suffix)
    retriever = BM25Retriever(corpus)
    retrieved_evidence: list[str] = []
    for claim in claims:
        hits = retriever.retrieve(claim, top_k=1)
        retrieved_evidence.append(hits[0].text if hits else "")
    retrieval_runtime = round(perf_counter() - started, 3)

    tokenizer, model = _load_transformer(args.checkpoint)
    model.eval()

    oracle_texts = [format_input(claim, evidence) for claim, evidence in zip(claims, oracle_evidence)]
    end_to_end_texts = [format_input(claim, evidence) for claim, evidence in zip(claims, retrieved_evidence)]

    oracle_started = perf_counter()
    oracle_preds = _predict(model, tokenizer, oracle_texts, args.max_length, args.batch_size)
    oracle_latency_ms = (perf_counter() - oracle_started) * 1000.0 / len(rows)

    e2e_started = perf_counter()
    e2e_preds = _predict(model, tokenizer, end_to_end_texts, args.max_length, args.batch_size)
    e2e_latency_ms = (perf_counter() - e2e_started) * 1000.0 / len(rows)

    y_true = [LABEL_TO_ID[label] for label in labels]

    oracle_metrics = _metrics(y_true, oracle_preds, oracle_latency_ms)
    e2e_metrics = _metrics(y_true, e2e_preds, e2e_latency_ms)

    oracle_top_errors = _top_errors(rows, claims, labels, oracle_evidence, oracle_preds, args.top_errors)
    e2e_top_errors = _top_errors(rows, claims, labels, retrieved_evidence, e2e_preds, args.top_errors)

    gap = {
        "accuracy_gap": round(oracle_metrics["accuracy"] - e2e_metrics["accuracy"], 4),
        "macro_f1_gap": round(oracle_metrics["macro_f1"] - e2e_metrics["macro_f1"], 4),
    }

    oracle_report = {
        "checkpoint": args.checkpoint,
        "evidence_source": "gold/curated evidence from data/processed/verifier_test.jsonl",
        "example_count": len(rows),
        **oracle_metrics,
        "top_errors": oracle_top_errors,
    }
    e2e_report = {
        "checkpoint": args.checkpoint,
        "evidence_source": (
            f"top-1 BM25 retrieval over data/processed/evidence_corpus{args.evidence_corpus_suffix}.jsonl"
            f" ({len(corpus)} passages)"
        ),
        "retrieval_runtime_seconds": retrieval_runtime,
        "example_count": len(rows),
        **e2e_metrics,
        "oracle_vs_retrieved_gap": gap,
        "top_errors": e2e_top_errors,
    }

    for path, report, title in (
        (Path(args.oracle_report_json), oracle_report, "Oracle"),
        (Path(args.end_to_end_report_json), e2e_report, "End-to-End (Retrieved)"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_report(report, path)

    Path(args.oracle_report_md).write_text(_to_markdown("Oracle Verifier Evaluation", oracle_report), encoding="utf-8")
    Path(args.end_to_end_report_md).write_text(
        _to_markdown("End-to-End Verifier Evaluation (Retrieved Evidence)", e2e_report), encoding="utf-8"
    )

    print(f"oracle: accuracy={oracle_metrics['accuracy']:.3f} macro_f1={oracle_metrics['macro_f1']:.3f}")
    print(f"end_to_end: accuracy={e2e_metrics['accuracy']:.3f} macro_f1={e2e_metrics['macro_f1']:.3f}")
    print(f"gap: {gap}")


def _load_transformer(checkpoint: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    return tokenizer, model


def _predict(model, tokenizer, texts: list[str], max_length: int, batch_size: int) -> list[int]:  # noqa: ANN001
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            logits = model(**encoded).logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
    return predictions


def _metrics(y_true: list[int], y_pred: list[int], latency_ms_per_example: float) -> dict[str, object]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_ORDER,
        labels=list(range(len(LABEL_ORDER))),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
            }
            for label in LABEL_ORDER
        },
        "confusion_matrix": matrix.tolist(),
        "latency_ms_per_example": float(latency_ms_per_example),
    }


def _top_errors(
    rows: list[dict],
    claims: list[str],
    labels: list[str],
    evidence: list[str],
    preds: list[int],
    limit: int,
) -> list[dict[str, object]]:
    errors = []
    for row, claim, true_label, ev, pred in zip(rows, claims, labels, evidence, preds):
        pred_label = LABEL_ORDER[pred]
        if pred_label == true_label:
            continue
        errors.append(
            {
                "claim_id": row.get("claim_id", ""),
                "claim": claim,
                "true_label": true_label,
                "predicted_label": pred_label,
                "evidence": ev[:300],
            }
        )
        if len(errors) >= limit:
            break
    return errors


def _to_markdown(title: str, report: dict[str, object]) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    lines.append(f"- Evidence source: {report['evidence_source']}")
    if "retrieval_runtime_seconds" in report:
        lines.append(f"- Retrieval runtime (s): {report['retrieval_runtime_seconds']}")
    lines.append(f"- Example count: {report['example_count']}")
    lines.append(f"- Latency (ms/example): {report['latency_ms_per_example']:.2f}")
    lines.append("")
    lines.append(f"- accuracy: {report['accuracy']:.3f}")
    lines.append(f"- macro_f1: {report['macro_f1']:.3f}")
    lines.append("")
    lines.append("## Per-class metrics")
    lines.append("")
    lines.append("| label | precision | recall | f1 |")
    lines.append("| --- | --- | --- | --- |")
    for label, metrics in report["per_class"].items():
        lines.append(f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |")
    lines.extend(["", "## Confusion matrix", ""])
    header = ["true \\ pred", *LABEL_ORDER]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for label, row in zip(LABEL_ORDER, report["confusion_matrix"]):
        lines.append("| " + " | ".join([label, *(str(value) for value in row)]) + " |")
    if "oracle_vs_retrieved_gap" in report:
        gap = report["oracle_vs_retrieved_gap"]
        lines.extend(["", "## Oracle vs. retrieved gap", ""])
        lines.append(f"- accuracy gap (oracle - end_to_end): {gap['accuracy_gap']:.4f}")
        lines.append(f"- macro_f1 gap (oracle - end_to_end): {gap['macro_f1_gap']:.4f}")
    if report.get("top_errors"):
        lines.extend(["", "## Top errors", ""])
        for error in report["top_errors"]:
            lines.append(
                f"- claim_id={error['claim_id']} true={error['true_label']} pred={error['predicted_label']}: "
                f"\"{error['claim']}\" | evidence: \"{error['evidence']}\""
            )
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
