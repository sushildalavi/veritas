"""Claim category error analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ErrorExample:
    claim: str
    truth: str
    prediction: str
    category: str


def categorize_claim(claim: str) -> str:
    lowered = claim.lower()
    if any(token in lowered for token in ["not ", "never", "no "]):
        return "negation"
    if any(char.isdigit() for char in claim):
        return "numerical"
    if any(month in lowered for month in [" january", " february", " march", " april", " may ", " june", " july", " august", " september", " october", " november", " december"]):
        return "temporal"
    if any(token in lowered for token in ["who", "where", "which", "what"]):
        return "entity"
    if any(token in lowered for token in ["because", "therefore", "as a result"]):
        return "possible multi-hop"
    return "entity"


def build_error_analysis(examples: list[ErrorExample]) -> dict[str, object]:
    counts = Counter(example.category for example in examples)
    return {"category_counts": dict(counts), "examples": [example.__dict__ for example in examples]}


def write_error_report(examples: list[ErrorExample], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "error_analysis.json"
    path.write_text(_json_dumps(build_error_analysis(examples)), encoding="utf-8")
    return path


def _json_dumps(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)
