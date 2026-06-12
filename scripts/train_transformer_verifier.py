"""Train a transformer-based claim verifier on the sampled datasets."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class TrainingExample:
    text: str
    label: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a transformer verifier checkpoint.")
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--train-file", default="data/processed/fever_train.jsonl")
    parser.add_argument("--val-file", default="data/processed/fever_val.jsonl")
    parser.add_argument("--test-file", default="data/processed/fever_test.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/transformer_verifier")
    parser.add_argument("--report-json", default="reports/transformer_verifier_eval.json")
    parser.add_argument("--report-md", default="reports/transformer_verifier_eval.md")
    parser.add_argument("--max-train-examples", type=int, default=20)
    parser.add_argument("--max-val-examples", type=int, default=10)
    parser.add_argument("--max-test-examples", type=int, default=10)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    try:
        report = train_transformer_verifier(
            model_name=args.model_name,
            train_file=Path(args.train_file),
            val_file=Path(args.val_file),
            test_file=Path(args.test_file),
            output_dir=output_dir,
            max_train_examples=args.max_train_examples,
            max_val_examples=args.max_val_examples,
            max_test_examples=args.max_test_examples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
        )
    except Exception as exc:  # pragma: no cover - failure path is surfaced explicitly
        failure_path = report_md.with_name(f"{report_md.stem}_FAILED.md")
        failure_path.write_text(
            "\n".join(
                [
                    "# Transformer Verifier Training Failed",
                    "",
                    f"- model_name: {args.model_name}",
                    f"- output_dir: {output_dir}",
                    "",
                    "The transformer verifier training could not complete.",
                    "",
                    f"Reason: {type(exc).__name__}: {exc}",
                    "",
                    "No metrics were fabricated.",
                ]
            ),
            encoding="utf-8",
        )
        raise SystemExit(f"Transformer verifier training failed: {exc}") from exc

    report["training_runtime_seconds"] = round(perf_counter() - started, 3)
    write_report(report, report_json)
    report_md.write_text(_to_markdown(report), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved transformer verifier checkpoint to {output_dir}")


def train_transformer_verifier(
    *,
    model_name: str,
    train_file: Path,
    val_file: Path,
    test_file: Path,
    output_dir: Path,
    max_train_examples: int,
    max_val_examples: int,
    max_test_examples: int,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)

    train_examples = _load_examples(train_file, max_examples=max_train_examples, seed=seed)
    val_examples = _load_examples(val_file, max_examples=max_val_examples, seed=seed + 1)
    test_examples = _load_examples(test_file, max_examples=max_test_examples, seed=seed + 2)
    if not train_examples:
        raise RuntimeError("no training examples were loaded")

    tokenizer, model = _load_transformer(model_name)
    train_dataset = _build_dataset(train_examples, tokenizer, max_length)
    val_dataset = _build_dataset(val_examples, tokenizer, max_length)
    test_dataset = _build_dataset(test_examples, tokenizer, max_length)

    trainer = _build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        output_dir=output_dir,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    train_metrics = _evaluate_dataset(trainer, train_dataset, train_examples)
    val_metrics = _evaluate_dataset(trainer, val_dataset, val_examples)
    test_metrics = _evaluate_dataset(trainer, test_dataset, test_examples)
    report = {
        "model_name": model_name,
        "checkpoint_path": str(output_dir),
        "label_order": LABEL_ORDER,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "train_example_count": len(train_examples),
        "validation_example_count": len(val_examples),
        "test_example_count": len(test_examples),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "test_per_class": test_metrics["per_class"],
        "test_latency_ms_per_example": test_metrics["latency_ms_per_example"],
    }
    return report


def _load_examples(path: Path, *, max_examples: int, seed: int) -> list[TrainingExample]:
    rows = read_jsonl(path)
    if max_examples > 0:
        rng = random.Random(seed)
        if len(rows) > max_examples:
            rows = rng.sample(rows, max_examples)
    examples: list[TrainingExample] = []
    for row in rows:
        claim = str(row.get("claim", ""))
        evidence = row.get("evidence") or []
        evidence_text = " ".join(str(item.get("text", "")) for item in evidence if item.get("text"))
        label = _normalize_label(str(row.get("label", "NOT_ENOUGH_INFO")))
        examples.append(TrainingExample(text=_format_input(claim, evidence_text), label=label))
    return examples


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def _format_input(claim: str, evidence_text: str) -> str:
    return f"Claim: {claim}\nEvidence: {evidence_text}"


def _load_transformer(model_name: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_ORDER),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    return tokenizer, model


def _build_dataset(examples: list[TrainingExample], tokenizer, max_length: int):  # noqa: ANN001
    if not examples:
        from datasets import Dataset  # type: ignore

        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
    from datasets import Dataset  # type: ignore

    label_ids = [LABEL_TO_ID[example.label] for example in examples]
    dataset = Dataset.from_dict({"text": [example.text for example in examples], "labels": label_ids})

    def _tokenize(batch):  # noqa: ANN001
        encoded = tokenizer(batch["text"], truncation=True, padding=False, max_length=max_length)
        encoded["labels"] = batch["labels"]
        return encoded

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return tokenized


def _build_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    batch_size: int,
    learning_rate: float,
    epochs: float,
    seed: int,
    output_dir: Path,
):  # noqa: ANN001
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments  # type: ignore

    def compute_metrics(eval_pred):  # noqa: ANN001
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
        }

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        seed=seed,
        data_seed=seed,
        logging_strategy="epoch",
        eval_strategy="epoch" if len(eval_dataset) else "no",
        save_strategy="no",
        report_to=[],
        use_cpu=True,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) else None,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics if len(eval_dataset) else None,
    )


def _evaluate_dataset(trainer, dataset, examples: list[TrainingExample]):  # noqa: ANN001
    if not examples:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "example_count": 0,
            "per_class": {},
            "confusion_matrix": [[0, 0, 0] for _ in LABEL_ORDER],
            "latency_ms_per_example": 0.0,
        }
    started = perf_counter()
    predictions = trainer.predict(dataset)
    latency_ms_per_example = (perf_counter() - started) * 1000.0 / max(len(examples), 1)
    y_true = [LABEL_TO_ID[example.label] for example in examples]
    y_pred = np.argmax(predictions.predictions, axis=1)
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
        "example_count": int(len(examples)),
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


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Transformer Verifier Evaluation",
        "",
        f"- Model: `{report['model_name']}`",
        f"- Checkpoint: `{report['checkpoint_path']}`",
        f"- Train examples: {report['train_example_count']}",
        f"- Validation examples: {report['validation_example_count']}",
        f"- Test examples: {report['test_example_count']}",
        f"- Test latency ms per example: {report['test_latency_ms_per_example']:.2f}",
        "",
        "| split | examples | accuracy | macro_f1 |",
        "| --- | --- | --- | --- |",
    ]
    for split_name in ("train", "validation", "test"):
        metrics = report[split_name]
        lines.append(
            f"| {split_name} | {int(metrics['example_count'])} | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} |"
        )
    lines.extend(["", "## Per-class test metrics", ""])
    lines.append("| label | precision | recall | f1 |")
    lines.append("| --- | --- | --- | --- |")
    for label, metrics in report["test_per_class"].items():
        lines.append(
            f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |"
        )
    lines.extend(["", "## Confusion matrix", "", "```text"])
    for row in report["test_confusion_matrix"]:
        lines.append(" ".join(str(value) for value in row))
    lines.append("```")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
