"""DeBERTa verifier fine-tuning script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from training.config import load_yaml
from training.losses import label_to_index


@dataclass(frozen=True)
class DebertaTrainingConfig:
    model_name: str = "microsoft/deberta-v3-base"
    output_dir: str = "checkpoints/deberta"
    max_length: int = 384
    train_split: str = "train"
    eval_split: str = "validation"


def load_dataset_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(__import__("json").loads(line))
    return rows


def load_config(path: str | Path) -> dict[str, object]:
    return load_yaml(path)


def prepare_examples(rows: list[dict]) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows:
        examples.append(
            {
                "text": f"claim: {row['claim']}\nevidence: {row.get('evidence_text', row.get('text', ''))}",
                "label": label_to_index(row["label"]),
            }
        )
    return examples


def main() -> None:  # pragma: no cover - script entrypoint
    print("Training script placeholder for offline DeBERTa fine-tuning.")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
