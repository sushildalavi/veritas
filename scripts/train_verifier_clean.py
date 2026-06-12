"""Train a TF-IDF + Logistic Regression baseline on the clean verifier dataset.

Reads ``data/processed/verifier_{train,val,test}.jsonl`` (built by
``scripts/build_verifier_dataset.py``), trains a TF-IDF + LogisticRegression
pipeline on ``Claim: <claim>\\nEvidence: <evidence>`` inputs, and writes:

  checkpoints/verifier_clean/model.joblib
  checkpoints/verifier_clean/metadata.json
  reports/verifier_clean_baseline.{json,md}
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl

LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


@dataclass(frozen=True)
class Example:
    claim_id: str
    text: str
    label: str
    claim: str
    evidence: str


def format_input(claim: str, evidence: str) -> str:
    return f"Claim: {claim}\nEvidence: {evidence}"


def load_examples(path: Path) -> list[Example]:
    return [
        Example(
            claim_id=str(row.get("claim_id", "")),
            text=format_input(row.get("claim", ""), row.get("evidence", "")),
            label=row.get("label", "NOT_ENOUGH_INFO"),
            claim=row.get("claim", ""),
            evidence=row.get("evidence", ""),
        )
        for row in read_jsonl(path)
    ]


def evaluate_split(pipeline: Pipeline, examples: list[Example]) -> dict[str, object]:
    if not examples:
        return {"example_count": 0, "accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion_matrix": []}

    texts = [example.text for example in examples]
    labels = [example.label for example in examples]
    predictions = pipeline.predict(texts)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=LABELS, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    matrix = confusion_matrix(labels, predictions, labels=LABELS).tolist()

    return {
        "example_count": len(examples),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": LABELS,
        "predictions": predictions.tolist(),
    }


def top_errors(examples: list[Example], predictions: list[str], *, limit: int = 20) -> list[dict[str, object]]:
    errors = [
        {
            "claim_id": example.claim_id,
            "claim": example.claim,
            "evidence": example.evidence[:200],
            "true_label": example.label,
            "predicted_label": prediction,
        }
        for example, prediction in zip(examples, predictions)
        if example.label != prediction
    ]
    return errors[:limit]


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def build_markdown(report: dict[str, object]) -> str:
    metadata = report.get("metadata", {})
    lines = [
        "# Verifier Clean Baseline",
        "",
        f"- Checkpoint: `{report['checkpoint_path']}`",
        f"- Model: TF-IDF (1-2 grams) + LogisticRegression",
        f"- Classes: {', '.join(LABELS)}",
        f"- Sklearn version: {metadata.get('sklearn_version', 'unknown')}",
        f"- Python version: {metadata.get('python_version', 'unknown')}",
        f"- Git commit: {metadata.get('git_commit', 'unknown')}",
        f"- Training command: `{metadata.get('training_command', 'unknown')}`",
        "",
        "| split | examples | accuracy | macro_f1 |",
        "| --- | --- | --- | --- |",
    ]
    for split_name in ("train", "validation", "test"):
        metrics = report[split_name]
        lines.append(
            f"| {split_name} | {metrics['example_count']} | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} |"
        )
    lines.append("")
    lines.append("## Test set per-class metrics")
    lines.append("")
    lines.append("| label | precision | recall | f1 | support |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label, metrics in report["test"]["per_class"].items():
        lines.append(
            f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} "
            f"| {metrics['f1']:.3f} | {metrics['support']} |"
        )
    lines.append("")
    lines.append("## Test set confusion matrix")
    lines.append("")
    header = ["true \\ pred", *LABELS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for label, row in zip(LABELS, report["test"]["confusion_matrix"]):
        lines.append("| " + " | ".join([label, *(str(value) for value in row)]) + " |")
    lines.append("")
    thresholds = report["thresholds"]
    lines.append("## Threshold check")
    lines.append("")
    lines.append(f"- macro_f1 >= 0.55: {'PASS' if thresholds['macro_f1_ok'] else 'FAIL'} ({report['test']['macro_f1']:.3f})")
    lines.append(f"- accuracy >= 0.60: {'PASS' if thresholds['accuracy_ok'] else 'FAIL'} ({report['test']['accuracy']:.3f})")
    lines.append("")
    lines.append("## Top test errors")
    lines.append("")
    lines.append("| claim_id | claim | evidence | true | predicted |")
    lines.append("| --- | --- | --- | --- | --- |")
    for error in report["top_test_errors"]:
        claim = error["claim"].replace("|", "\\|")
        evidence = error["evidence"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {error['claim_id']} | {claim} | {evidence} | {error['true_label']} | {error['predicted_label']} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the clean sklearn verifier baseline.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="checkpoints/verifier_clean")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=10000)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    reports_dir = Path(args.reports_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_examples = load_examples(data_dir / "verifier_train.jsonl")
    val_examples = load_examples(data_dir / "verifier_val.jsonl")
    test_examples = load_examples(data_dir / "verifier_test.jsonl")

    if not train_examples:
        raise RuntimeError("no training examples found; run scripts/build_verifier_dataset.py first")

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=args.max_features)),
            ("clf", LogisticRegression(max_iter=1000, random_state=args.seed, class_weight="balanced")),
        ]
    )
    pipeline.fit([example.text for example in train_examples], [example.label for example in train_examples])

    checkpoint_path = checkpoint_dir / "model.joblib"
    joblib.dump({"pipeline": pipeline, "label_order": LABELS}, checkpoint_path)

    train_report = evaluate_split(pipeline, train_examples)
    val_report = evaluate_split(pipeline, val_examples)
    test_report = evaluate_split(pipeline, test_examples)

    test_predictions = test_report.pop("predictions")
    train_report.pop("predictions", None)
    val_report.pop("predictions", None)

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "training_command": shlex.join([Path(sys.executable).name, "scripts/train_verifier_clean.py", *sys.argv[1:]]),
        "seed": args.seed,
        "data_dir": str(data_dir),
        "sample_sizes": {
            "train_examples": len(train_examples),
            "validation_examples": len(val_examples),
            "test_examples": len(test_examples),
        },
    }

    thresholds = {
        "macro_f1_ok": test_report["macro_f1"] >= 0.55,
        "accuracy_ok": test_report["accuracy"] >= 0.60,
    }

    report = {
        "checkpoint_path": str(checkpoint_path),
        "train": train_report,
        "validation": val_report,
        "test": test_report,
        "thresholds": thresholds,
        "top_test_errors": top_errors(test_examples, test_predictions),
        "metadata": metadata,
    }

    write_report(report, reports_dir / "verifier_clean_baseline.json")
    (reports_dir / "verifier_clean_baseline.md").write_text(build_markdown(report), encoding="utf-8")

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"test accuracy={test_report['accuracy']:.3f} macro_f1={test_report['macro_f1']:.3f}")
    print(f"thresholds: {thresholds}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
