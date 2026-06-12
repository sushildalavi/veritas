"""Train a lightweight claim verifier checkpoint from the sampled datasets."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_records


@dataclass(frozen=True)
class TrainingExample:
    text: str
    label: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight verifier checkpoint.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="checkpoints/verifier")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    reports_dir = Path(args.reports_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records_by_split = load_records(data_dir)
    train_records = list(records_by_split.get("fever_train", [])) + list(records_by_split.get("scifact_train", []))
    val_records = list(records_by_split.get("fever_val", [])) + list(records_by_split.get("scifact_val", []))
    test_records = list(records_by_split.get("fever_test", [])) + list(records_by_split.get("scifact_test", []))

    train_examples = _build_examples(train_records, seed=args.seed)
    val_examples = _build_examples(val_records, seed=args.seed + 1)
    test_examples = _build_examples(test_records, seed=args.seed + 2)

    if not train_examples:
        raise RuntimeError("no training examples were built from the processed data")

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=500, random_state=args.seed)),
        ]
    )
    pipeline.fit([example.text for example in train_examples], [example.label for example in train_examples])

    checkpoint_path = checkpoint_dir / "model.joblib"
    joblib.dump({"pipeline": pipeline, "label_order": list(pipeline.classes_)}, checkpoint_path)

    report = {
        "checkpoint_path": str(checkpoint_path),
        "train": _evaluate_split(pipeline, train_examples),
        "validation": _evaluate_split(pipeline, val_examples),
        "test": _evaluate_split(pipeline, test_examples),
        "class_labels": list(pipeline.classes_),
        "train_example_count": len(train_examples),
        "validation_example_count": len(val_examples),
        "test_example_count": len(test_examples),
    }
    write_report(report, reports_dir / "verifier_training.json")
    (reports_dir / "verifier_training.md").write_text(_to_markdown(report), encoding="utf-8")
    (checkpoint_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved verifier checkpoint to {checkpoint_path}")


def _build_examples(records, *, seed: int) -> list[TrainingExample]:
    rng = random.Random(seed)
    evidence_pool = [span.text for record in records for span in record.evidence if span.text]
    examples: list[TrainingExample] = []
    for record in records:
        positive_spans = [span for span in record.evidence if span.text]
        if positive_spans:
            for span in positive_spans:
                examples.append(TrainingExample(text=_format_input(record.claim, span.text), label=_normalize_label(record.label)))
            if _normalize_label(record.label) != "NOT ENOUGH INFO" and evidence_pool:
                negative_text = _sample_negative_text(rng, evidence_pool, positive_spans)
                examples.append(TrainingExample(text=_format_input(record.claim, negative_text), label="NOT ENOUGH INFO"))
        else:
            examples.append(TrainingExample(text=_format_input(record.claim, ""), label="NOT ENOUGH INFO"))
    return examples


def _sample_negative_text(rng: random.Random, evidence_pool: list[str], positive_spans: list[EvidenceSpan]) -> str:
    positive_texts = {span.text for span in positive_spans}
    candidates = [text for text in evidence_pool if text not in positive_texts]
    if not candidates:
        candidates = evidence_pool
    return rng.choice(candidates) if candidates else ""


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def _format_input(claim: str, evidence_text: str) -> str:
    return f"claim: {claim}\nevidence: {evidence_text}"


def _evaluate_split(pipeline: Pipeline, examples: list[TrainingExample]) -> dict[str, float]:
    if not examples:
        return {"accuracy": 0.0, "macro_f1": 0.0, "example_count": 0.0}
    predictions = pipeline.predict([example.text for example in examples])
    labels = [example.label for example in examples]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "example_count": float(len(examples)),
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Verifier Training",
        "",
        f"- Checkpoint: `{report['checkpoint_path']}`",
        f"- Classes: {', '.join(report['class_labels'])}",
        "",
        "| split | examples | accuracy | macro_f1 |",
        "| --- | --- | --- | --- |",
    ]
    for split_name in ("train", "validation", "test"):
        metrics = report[split_name]
        lines.append(
            f"| {split_name} | {int(metrics['example_count'])} | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
