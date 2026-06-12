"""DPO offline alignment script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Any

from models.dpo_model import DPOModelConfig, load_dpo_model
from training.config import load_yaml


@dataclass(frozen=True)
class DPOTrainingConfig:
    base_model: str = "microsoft/phi-2"
    output_dir: str = "checkpoints/dpo"
    beta: float = 0.1


def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def build_training_artifacts(config: DPOTrainingConfig) -> dict[str, object]:
    return {"config": config, "model": load_dpo_model(DPOModelConfig(base_model=config.base_model))}


def build_dpo_trainer(*args: Any, **kwargs: Any):  # pragma: no cover - optional dependency
    try:
        from trl import DPOTrainer  # type: ignore

        return DPOTrainer(*args, **kwargs)
    except Exception:
        return None


def main() -> None:  # pragma: no cover - script entrypoint
    parser = argparse.ArgumentParser(description="Run the offline DPO training pipeline.")
    parser.add_argument("--config", default="configs/dpo.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Loaded DPO config for base model {config.get('base_model', 'unknown')}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
