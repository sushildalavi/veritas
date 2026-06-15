"""Train a verifier on a mix of gold and retrieved evidence.

Builds a training set that mixes:

  - gold evidence examples (data/processed/verifier_train.jsonl)
  - retrieved top-1/top-3/top-5 evidence examples, reranked by the fine-tuned
    cross-encoder (data/processed/verifier_train_topk_augmented.jsonl)
  - hard-negative NOT_ENOUGH_INFO examples: each claim paired with a different
    claim's gold evidence, so the verifier learns to abstain when the
    retrieved evidence does not actually address the claim

Model selection uses validation macro-F1 on the gold validation set, with
early stopping, so the result is directly comparable with
``checkpoints/transformer_verifier_clean``.

Reads:
  data/processed/verifier_{train,val,test}.jsonl
  data/processed/verifier_{train,val,test}_topk_augmented.jsonl

Writes:
  checkpoints/retrieved_augmented_verifier/
  reports/retrieved_augmented_verifier_eval.{json,md}
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import shlex
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from train_transformer_verifier_clean import (
    Example,
    _build_dataset,
    _build_trainer,
    _class_weights,
    _evaluate_dataset,
    _git_commit_hash,
    _load_transformer,
    load_examples,
)

TOPK_FIELDS = ["top1_evidence", "top3_evidence", "top5_evidence"]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_mixed_train_examples(gold_path: Path, topk_path: Path, *, seed: int) -> list[Example]:
    examples = list(load_examples(gold_path))
    topk_rows = read_jsonl(topk_path)

    rng = random.Random(seed)
    for row in topk_rows:
        claim = str(row["claim"])
        label = row["label"]
        for field in TOPK_FIELDS:
            evidence = str(row.get(field, "")).strip()
            if evidence:
                examples.append(Example(claim_id=str(row["claim_id"]), claim=claim, evidence=evidence, label=label))

        if len(topk_rows) > 1:
            distractor = row
            while distractor["claim_id"] == row["claim_id"]:
                distractor = topk_rows[rng.randrange(len(topk_rows))]

            # Hard-negative NEI #1: claim + a different claim's gold evidence,
            # unformatted (matches the gold-evidence input style).
            distractor_gold = str(distractor.get("gold_evidence", "")).strip()
            if distractor_gold:
                examples.append(
                    Example(claim_id=str(row["claim_id"]), claim=claim, evidence=distractor_gold, label="NOT_ENOUGH_INFO")
                )

            # Hard-negative NEI #2: claim + a different claim's retrieved
            # "[E1] ..." formatted evidence, so NEI is not spuriously tied to
            # the absence of the [E1]-style retrieval formatting.
            distractor_retrieved = str(distractor.get("top1_evidence", "")).strip()
            if distractor_retrieved:
                examples.append(
                    Example(claim_id=str(row["claim_id"]), claim=claim, evidence=distractor_retrieved, label="NOT_ENOUGH_INFO")
                )

    return examples


def examples_from_topk(rows: list[dict], field: str) -> list[Example]:
    return [
        Example(claim_id=str(row["claim_id"]), claim=str(row["claim"]), evidence=str(row.get(field, "")), label=row["label"])
        for row in rows
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a verifier on mixed gold/retrieved/hard-negative evidence.")
    parser.add_argument("--config", default="configs/retrieved_augmented_verifier.yaml")
    parser.add_argument("--report-json", default="reports/retrieved_augmented_verifier_eval.json")
    parser.add_argument("--report-md", default="reports/retrieved_augmented_verifier_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    checkpoint_cfg = config["checkpoint"]

    output_dir = Path(checkpoint_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = train_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)

    train_examples = build_mixed_train_examples(
        Path(data_cfg["gold_train_file"]), Path(data_cfg["topk_train_file"]), seed=seed
    )
    val_examples = load_examples(Path(data_cfg["gold_val_file"]))
    test_examples = load_examples(Path(data_cfg["gold_test_file"]))
    topk_test_rows = read_jsonl(Path(data_cfg["topk_test_file"]))

    max_length = train_cfg["max_length"]
    tokenizer, model = _load_transformer(model_cfg["base_model"])
    train_dataset = _build_dataset(train_examples, tokenizer, max_length)
    val_dataset = _build_dataset(val_examples, tokenizer, max_length)
    test_dataset = _build_dataset(test_examples, tokenizer, max_length)
    class_weights = _class_weights(train_examples)

    from transformers import EarlyStoppingCallback  # type: ignore

    trainer = _build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        epochs=train_cfg["epochs"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        use_class_weights=train_cfg["use_class_weights"],
        use_weighted_sampler=train_cfg["use_weighted_sampler"],
        use_cpu=True,
        seed=seed,
        output_dir=output_dir,
        class_weights=class_weights,
        train_examples=train_examples,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"])],
    )

    started = perf_counter()
    trainer.train()
    runtime_seconds = perf_counter() - started
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    trainer.args.dataloader_drop_last = False

    val_metrics = _evaluate_dataset(trainer, val_dataset, val_examples)
    oracle_metrics = _evaluate_dataset(trainer, test_dataset, test_examples)

    topk_metrics: dict[str, dict] = {}
    for name, field in (("top1", "top1_evidence"), ("top3", "top3_evidence"), ("top5", "top5_evidence")):
        topk_examples = examples_from_topk(topk_test_rows, field)
        topk_dataset = _build_dataset(topk_examples, tokenizer, max_length)
        topk_metrics[name] = _evaluate_dataset(trainer, topk_dataset, topk_examples)

    gaps = {
        name: {
            "accuracy_gap": round(oracle_metrics["accuracy"] - metrics["accuracy"], 4),
            "macro_f1_gap": round(oracle_metrics["macro_f1"] - metrics["macro_f1"], 4),
        }
        for name, metrics in topk_metrics.items()
    }

    report = {
        "model_name": model_cfg["base_model"],
        "checkpoint_path": str(output_dir),
        "train_example_count": len(train_examples),
        "training": train_cfg,
        "training_runtime_seconds": round(runtime_seconds, 2),
        "validation": val_metrics,
        "oracle": oracle_metrics,
        "top_k": topk_metrics,
        "oracle_vs_retrieved_gap": gaps,
        "thresholds": {
            "macro_f1_ok": oracle_metrics["macro_f1"] >= 0.55,
            "accuracy_ok": oracle_metrics["accuracy"] >= 0.60,
        },
        "metadata": {
            "python_version": platform.python_version(),
            "git_commit": _git_commit_hash(),
            "training_command": shlex.join([Path(sys.executable).name, "scripts/train_retrieved_augmented_verifier.py", *sys.argv[1:]]),
            "seed": seed,
        },
    }
    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved retrieved-augmented verifier checkpoint to {output_dir}")
    print(
        f"oracle macro_f1={oracle_metrics['macro_f1']:.3f}, "
        f"top1={topk_metrics['top1']['macro_f1']:.3f}, "
        f"top3={topk_metrics['top3']['macro_f1']:.3f}, "
        f"top5={topk_metrics['top5']['macro_f1']:.3f}"
    )


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Retrieved-Augmented Verifier Evaluation",
        "",
        f"- model_name: {report['model_name']}",
        f"- checkpoint_path: {report['checkpoint_path']}",
        f"- train_example_count: {report['train_example_count']}",
        f"- training_runtime_seconds: {report['training_runtime_seconds']}",
        "",
        "| split | examples | accuracy | macro_f1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.append(
        f"| validation (gold) | {report['validation']['example_count']} | "
        f"{report['validation']['accuracy']:.3f} | {report['validation']['macro_f1']:.3f} |"
    )
    lines.append(
        f"| test oracle (gold) | {report['oracle']['example_count']} | "
        f"{report['oracle']['accuracy']:.3f} | {report['oracle']['macro_f1']:.3f} |"
    )
    for name, metrics in report["top_k"].items():
        lines.append(f"| test {name} (retrieved) | {metrics['example_count']} | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} |")

    lines += ["", "## Oracle vs. retrieved gap (test set)", ""]
    for name, gap in report["oracle_vs_retrieved_gap"].items():
        lines.append(f"- {name}: accuracy_gap={gap['accuracy_gap']} macro_f1_gap={gap['macro_f1_gap']}")

    lines += ["", "## Test oracle per-class metrics", ""]
    lines.append("| label | precision | recall | f1 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for label, metrics in report["oracle"]["per_class"].items():
        lines.append(f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |")

    thresholds = report["thresholds"]
    lines += ["", "## Threshold check", ""]
    lines.append(f"- oracle macro_f1 >= 0.55: {'PASS' if thresholds['macro_f1_ok'] else 'FAIL'} ({report['oracle']['macro_f1']:.3f})")
    lines.append(f"- oracle accuracy >= 0.60: {'PASS' if thresholds['accuracy_ok'] else 'FAIL'} ({report['oracle']['accuracy']:.3f})")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
