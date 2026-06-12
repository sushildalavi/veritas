"""DPO-aligned explanation model stub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DPOModelConfig:
    base_model: str = "microsoft/phi-2"
    adapter_path: str | Path | None = None


def load_dpo_model(config: DPOModelConfig):
    return {
        "base_model": config.base_model,
        "adapter_path": str(config.adapter_path) if config.adapter_path else None,
        "available": False,
    }
