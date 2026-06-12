"""BEIR benchmark wrapper for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeirBenchmarkResult:
    dataset: str
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    status: str = "todo"


def run_beir_benchmark(dataset_name: str, *, output_dir: str | Path | None = None) -> BeirBenchmarkResult:
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    return BeirBenchmarkResult(dataset=dataset_name, recall_at_k={}, ndcg_at_k={})
