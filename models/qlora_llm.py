"""QLoRA model setup for offline alignment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QLoRAConfig:
    base_model: str = "microsoft/phi-2"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


def build_peft_config(config: QLoRAConfig):
    try:  # pragma: no cover - optional dependency
        from peft import LoraConfig, TaskType

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
        )
    except Exception:
        return config


def load_qlora_model(model_name: str | None = None, adapter_path: str | Path | None = None):
    return {
        "base_model": model_name or QLoRAConfig.base_model,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "available": False,
    }
