"""Pareto analysis of quality versus deployment costs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    model: str
    macro_f1: float
    latency_ms: float
    memory_mb: float
    deployment_feasibility: float


def pareto_report(points: list[ParetoPoint]) -> list[dict[str, object]]:
    return [
        {
            "model": point.model,
            "macro_f1": point.macro_f1,
            "latency_ms": point.latency_ms,
            "memory_mb": point.memory_mb,
            "deployment_feasibility": point.deployment_feasibility,
        }
        for point in sorted(points, key=lambda point: (-point.macro_f1, point.latency_ms, point.memory_mb))
    ]
