"""QLoRA offline training script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.qlora_llm import QLoRAConfig, build_peft_config
from training.config import load_yaml


@dataclass(frozen=True)
class QLoRATrainingConfig:
    base_model: str = "microsoft/phi-2"
    output_dir: str = "checkpoints/qlora_phi"
    max_seq_length: int = 1024


def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def build_training_artifacts(config: QLoRATrainingConfig) -> dict[str, object]:
    peft_config = build_peft_config(QLoRAConfig(base_model=config.base_model))
    return {"config": config, "peft_config": peft_config}


def main() -> None:  # pragma: no cover - script entrypoint
    print("QLoRA offline training script placeholder.")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
