"""Fine-tune a sentence-transformers bi-encoder retriever on claim/evidence pairs.

Reads ``data/processed/retriever_train_pairs.jsonl`` (built by
``scripts/build_evidence_pair_dataset.py``) and fine-tunes a bi-encoder with
``MultipleNegativesRankingLoss`` using the claim as the anchor, the gold
evidence as the positive, and mined hard negatives as in-batch negatives.

Writes:
  checkpoints/biencoder_retriever/
  reports/biencoder_retriever_train.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import yaml
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_dataset(pairs_path: Path, max_hard_negatives: int) -> Dataset:
    negative_cols = [f"negative_{i + 1}" for i in range(max_hard_negatives)]
    columns: dict[str, list[str]] = {"anchor": [], "positive": []}
    for col in negative_cols:
        columns[col] = []

    for row in read_jsonl(pairs_path):
        claim = row["claim"].strip()
        positive = row["positive_evidence"].strip()
        negatives = [text.strip() for text in row.get("hard_negatives", []) if text.strip()]
        if not claim or not positive or len(negatives) < max_hard_negatives:
            continue
        columns["anchor"].append(claim)
        columns["positive"].append(positive)
        for col, negative in zip(negative_cols, negatives[:max_hard_negatives]):
            columns[col].append(negative)

    return Dataset.from_dict(columns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder retriever on claim/evidence pairs.")
    parser.add_argument("--config", default="configs/biencoder_retriever.yaml")
    parser.add_argument("--report-json", default="reports/biencoder_retriever_train.json")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    model_cfg = config["model"]
    train_cfg = config["training"]
    checkpoint_cfg = config["checkpoint"]

    output_dir = Path(checkpoint_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(Path(train_cfg["train_file"]), train_cfg["max_hard_negatives"])

    started = perf_counter()
    model = SentenceTransformer(model_cfg["base_model"])
    loss = losses.MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "trainer_state"),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        seed=train_cfg["seed"],
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        use_cpu=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    model.save(str(output_dir))
    runtime_seconds = perf_counter() - started

    report = {
        "base_model": model_cfg["base_model"],
        "train_pairs": len(train_dataset),
        "num_epochs": train_cfg["num_epochs"],
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
        "max_hard_negatives": train_cfg["max_hard_negatives"],
        "runtime_seconds": round(runtime_seconds, 2),
        "output_dir": str(output_dir),
    }
    write_report(report, args.report_json)
    print(f"Trained bi-encoder retriever on {len(train_dataset)} pairs in {runtime_seconds:.1f}s -> {output_dir}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
