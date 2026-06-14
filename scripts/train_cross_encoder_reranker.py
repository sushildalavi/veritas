"""Fine-tune a cross-encoder reranker on claim/evidence relevance pairs.

Reads ``data/processed/reranker_train_pairs.jsonl`` (built by
``scripts/build_evidence_pair_dataset.py``) and fine-tunes a cross-encoder with
binary cross-entropy on (claim, evidence) -> relevance (1 for gold/positive,
0 for hard negatives).

Writes:
  checkpoints/cross_encoder_reranker/
  reports/cross_encoder_reranker_train.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import yaml
from datasets import Dataset
from sentence_transformers import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder import losses

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_dataset(pairs_path: Path) -> Dataset:
    sentence_1: list[str] = []
    sentence_2: list[str] = []
    labels: list[float] = []
    for row in read_jsonl(pairs_path):
        claim = row["claim"].strip()
        evidence = row["evidence"].strip()
        if not claim or not evidence:
            continue
        sentence_1.append(claim)
        sentence_2.append(evidence)
        labels.append(float(row["relevance"]))
    return Dataset.from_dict({"sentence_1": sentence_1, "sentence_2": sentence_2, "label": labels})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a cross-encoder reranker on claim/evidence pairs.")
    parser.add_argument("--config", default="configs/cross_encoder_reranker.yaml")
    parser.add_argument("--report-json", default="reports/cross_encoder_reranker_train.json")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    model_cfg = config["model"]
    train_cfg = config["training"]
    checkpoint_cfg = config["checkpoint"]

    output_dir = Path(checkpoint_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(Path(train_cfg["train_file"]))

    started = perf_counter()
    model = CrossEncoder(model_cfg["base_model"], max_length=train_cfg["max_length"])
    loss = losses.BinaryCrossEntropyLoss(model)

    training_args = CrossEncoderTrainingArguments(
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

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    runtime_seconds = perf_counter() - started

    report = {
        "base_model": model_cfg["base_model"],
        "train_pairs": len(train_dataset),
        "num_epochs": train_cfg["num_epochs"],
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
        "max_length": train_cfg["max_length"],
        "runtime_seconds": round(runtime_seconds, 2),
        "output_dir": str(output_dir),
    }
    write_report(report, args.report_json)
    print(f"Trained cross-encoder reranker on {len(train_dataset)} pairs in {runtime_seconds:.1f}s -> {output_dir}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
