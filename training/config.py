"""Shared configuration loading helpers for training scripts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def dataclass_to_dict(config: object) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    raise TypeError("config must be a dataclass instance")
