"""Fine-tune a transformer verifier on the clean verifier dataset.

Reads ``data/processed/verifier_{train,val,test}.jsonl`` (built by
``scripts/build_verifier_dataset.py``), fine-tunes a sequence classification
model on ``Claim: <claim>\\nEvidence: <evidence>`` inputs, and writes:

  checkpoints/transformer_verifier_clean/
  reports/transformer_verifier_clean_eval.{json,md}
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class Example:
    claim_id: str
    claim: str
    evidence: str
    label: str


def load_examples(path: Path) -> list[Example]:
    return [
        Example(
            claim_id=str(row.get("claim_id", "")),
            claim=str(row.get("claim", "")),
            evidence=str(row.get("evidence", "")),
            label=row.get("label", "NOT_ENOUGH_INFO"),
        )
        for row in read_jsonl(path)
    ]


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer verifier on the clean dataset.")
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="checkpoints/transformer_verifier_clean")
    parser.add_argument("--report-json", default="reports/transformer_verifier_clean_eval.json")
    parser.add_argument("--report-md", default="reports/transformer_verifier_clean_eval.md")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
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
            train_file=data_dir / "verifier_train.jsonl",
            val_file=data_dir / "verifier_val.jsonl",
            test_file=data_dir / "verifier_test.jsonl",
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            use_class_weights=args.use_class_weights,
            use_weighted_sampler=args.use_weighted_sampler,
            use_cpu=args.use_cpu,
            seed=args.seed,
        )
    except Exception as exc:  # pragma: no cover - failure path is surfaced explicitly
        failure_path = report_md.with_name(f"{report_md.stem}_FAILED.md")
        failure_path.write_text(
            "\n".join(
                [
                    "# Transformer Verifier (Clean) Training Failed",
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
    report["metadata"] = {
        "python_version": platform.python_version(),
        "git_commit": _git_commit_hash(),
        "training_command": shlex.join([Path(sys.executable).name, "scripts/train_transformer_verifier_clean.py", *sys.argv[1:]]),
        "seed": args.seed,
    }
    write_report(report, report_json)
    report_md.write_text(_to_markdown(report), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved transformer verifier checkpoint to {output_dir}")
    print(f"test accuracy={report['test']['accuracy']:.3f} macro_f1={report['test']['macro_f1']:.3f}")
    print(f"thresholds: {report['thresholds']}")


def train_transformer_verifier(
    *,
    model_name: str,
    train_file: Path,
    val_file: Path,
    test_file: Path,
    output_dir: Path,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    use_class_weights: bool,
    use_weighted_sampler: bool,
    use_cpu: bool,
    seed: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)

    train_examples = load_examples(train_file)
    val_examples = load_examples(val_file)
    test_examples = load_examples(test_file)
    if not train_examples:
        raise RuntimeError("no training examples were loaded; run scripts/build_verifier_dataset.py first")

    tokenizer, model = _load_transformer(model_name)
    train_dataset = _build_dataset(train_examples, tokenizer, max_length)
    val_dataset = _build_dataset(val_examples, tokenizer, max_length)
    test_dataset = _build_dataset(test_examples, tokenizer, max_length)
    class_weights = _class_weights(train_examples)

    trainer = _build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        use_class_weights=use_class_weights,
        use_weighted_sampler=use_weighted_sampler,
        use_cpu=use_cpu,
        seed=seed,
        output_dir=output_dir,
        class_weights=class_weights,
        train_examples=train_examples,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    train_metrics = _evaluate_dataset(trainer, train_dataset, train_examples)
    val_metrics = _evaluate_dataset(trainer, val_dataset, val_examples)
    test_metrics = _evaluate_dataset(trainer, test_dataset, test_examples)

    thresholds = {
        "macro_f1_ok": test_metrics["macro_f1"] >= 0.55,
        "accuracy_ok": test_metrics["accuracy"] >= 0.60,
    }

    return {
        "model_name": model_name,
        "checkpoint_path": str(output_dir),
        "label_order": LABEL_ORDER,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "thresholds": thresholds,
    }


def _class_weights(examples: list[Example]) -> list[float]:
    counts = [0] * len(LABEL_ORDER)
    for example in examples:
        counts[LABEL_TO_ID[_normalize_label(example.label)]] += 1
    total = sum(counts)
    return [total / (len(LABEL_ORDER) * count) for count in counts]


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


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


def _build_dataset(examples: list[Example], tokenizer, max_length: int):  # noqa: ANN001
    from datasets import Dataset  # type: ignore

    if not examples:
        return Dataset.from_dict({"claim": [], "evidence": [], "labels": []})

    label_ids = [LABEL_TO_ID[_normalize_label(example.label)] for example in examples]
    dataset = Dataset.from_dict(
        {
            "claim": [example.claim for example in examples],
            "evidence": [example.evidence for example in examples],
            "labels": label_ids,
        }
    )

    def _tokenize(batch):  # noqa: ANN001
        encoded = tokenizer(
            batch["claim"],
            batch["evidence"],
            truncation="only_second",
            padding=False,
            max_length=max_length,
        )
        encoded["labels"] = batch["labels"]
        return encoded

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["claim", "evidence"])
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
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    use_class_weights: bool,
    use_weighted_sampler: bool,
    use_cpu: bool,
    seed: int,
    output_dir: Path,
    class_weights: list[float],
    train_examples: list[Example],
):  # noqa: ANN001
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments  # type: ignore

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    label_counts = {label: 0 for label in LABEL_ORDER}
    for example in train_examples:
        label_counts[_normalize_label(example.label)] += 1
    example_weights = [
        1.0 / max(label_counts[_normalize_label(example.label)], 1)
        for example in train_examples
    ]

    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # noqa: ANN001
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                weight=class_weights_tensor.to(device=logits.device, dtype=logits.dtype),
            )
            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):  # noqa: ANN001
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            if not use_weighted_sampler:
                return super().get_train_dataloader()
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(seed)
            sampler = WeightedRandomSampler(
                weights=example_weights,
                num_samples=len(example_weights),
                replacement=True,
                generator=sampler_generator,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_num_workers > 0,
            )

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
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type="cosine",
        seed=seed,
        data_seed=seed,
        logging_strategy="epoch",
        eval_strategy="epoch" if len(eval_dataset) else "no",
        save_strategy="epoch" if len(eval_dataset) else "no",
        load_best_model_at_end=bool(len(eval_dataset)),
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,
        use_cpu=use_cpu,
        report_to=[],
    )
    trainer_cls = WeightedLossTrainer if use_class_weights else Trainer
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset if len(eval_dataset) else None,
        "processing_class": tokenizer,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics if len(eval_dataset) else None,
    }
    if use_class_weights:
        return trainer_cls(**trainer_kwargs)
    return trainer_cls(**trainer_kwargs)


def _evaluate_dataset(trainer, dataset, examples: list[Example]):  # noqa: ANN001
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
    y_true = [LABEL_TO_ID[_normalize_label(example.label)] for example in examples]
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
    metadata = report.get("metadata", {})
    lines = [
        "# Transformer Verifier (Clean Dataset) Evaluation",
        "",
        f"- Model: `{report['model_name']}`",
        f"- Checkpoint: `{report['checkpoint_path']}`",
        f"- Epochs: {report['epochs']}",
        f"- Batch size: {report['batch_size']}",
        f"- Learning rate: {report['learning_rate']}",
        f"- Training runtime (s): {report['training_runtime_seconds']}",
        f"- Git commit: {metadata.get('git_commit', 'unknown')}",
        f"- Training command: `{metadata.get('training_command', 'unknown')}`",
        "",
        "| split | examples | accuracy | macro_f1 |",
        "| --- | --- | --- | --- |",
    ]
    for split_name in ("train", "validation", "test"):
        metrics = report[split_name]
        lines.append(
            f"| {split_name} | {int(metrics['example_count'])} | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} |"
        )
    lines.extend(["", "## Test set per-class metrics", ""])
    lines.append("| label | precision | recall | f1 |")
    lines.append("| --- | --- | --- | --- |")
    for label, metrics in report["test"]["per_class"].items():
        lines.append(
            f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |"
        )
    lines.extend(["", "## Test set confusion matrix", ""])
    header = ["true \\ pred", *LABEL_ORDER]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for label, row in zip(LABEL_ORDER, report["test"]["confusion_matrix"]):
        lines.append("| " + " | ".join([label, *(str(value) for value in row)]) + " |")
    lines.append("")
    thresholds = report["thresholds"]
    lines.append("## Threshold check")
    lines.append("")
    lines.append(f"- macro_f1 >= 0.55: {'PASS' if thresholds['macro_f1_ok'] else 'FAIL'} ({report['test']['macro_f1']:.3f})")
    lines.append(f"- accuracy >= 0.60: {'PASS' if thresholds['accuracy_ok'] else 'FAIL'} ({report['test']['accuracy']:.3f})")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
