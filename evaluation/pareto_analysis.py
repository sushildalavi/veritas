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


def pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    frontier: list[ParetoPoint] = []
    for point in points:
        if any(_dominates(other, point) for other in points if other is not point):
            continue
        frontier.append(point)
    return sorted(frontier, key=lambda point: (-point.macro_f1, point.latency_ms, point.memory_mb))


def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
    better_or_equal = (
        left.macro_f1 >= right.macro_f1
        and left.latency_ms <= right.latency_ms
        and left.memory_mb <= right.memory_mb
        and left.deployment_feasibility >= right.deployment_feasibility
    )
    strictly_better = (
        left.macro_f1 > right.macro_f1
        or left.latency_ms < right.latency_ms
        or left.memory_mb < right.memory_mb
        or left.deployment_feasibility > right.deployment_feasibility
    )
    return better_or_equal and strictly_better
