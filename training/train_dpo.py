"""DPO offline alignment script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from models.dpo_model import DPOModelConfig, load_dpo_model


@dataclass(frozen=True)
class DPOTrainingConfig:
    base_model: str = "microsoft/phi-2"
    output_dir: str = "checkpoints/dpo"
    beta: float = 0.1


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_training_artifacts(config: DPOTrainingConfig) -> dict[str, object]:
    return {"config": config, "model": load_dpo_model(DPOModelConfig(base_model=config.base_model))}


def build_dpo_trainer(*args: Any, **kwargs: Any):  # pragma: no cover - optional dependency
    try:
        from trl import DPOTrainer  # type: ignore

        return DPOTrainer(*args, **kwargs)
    except Exception:
        return None


def main() -> None:  # pragma: no cover - script entrypoint
    print("DPO offline training script placeholder.")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
