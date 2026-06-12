"""A/B simulation with bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class BootstrapSummary:
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float


def bootstrap_ab_simulation(
    system_a: list[float],
    system_b: list[float],
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapSummary:
    if len(system_a) != len(system_b):
        raise ValueError("system_a and system_b must have the same length")
    if not system_a:
        return BootstrapSummary(0.0, 0.0, 0.0, 0.0, 0.0)

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(len(system_a)) for _ in range(len(system_a))]
        sample_a = [system_a[index] for index in indices]
        sample_b = [system_b[index] for index in indices]
        deltas.append((sum(sample_b) / len(sample_b)) - (sum(sample_a) / len(sample_a)))

    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = int(tail * len(deltas))
    high_index = min(len(deltas) - 1, int((1.0 - tail) * len(deltas)) - 1)
    return BootstrapSummary(
        mean_a=sum(system_a) / len(system_a),
        mean_b=sum(system_b) / len(system_b),
        delta=(sum(system_b) / len(system_b)) - (sum(system_a) / len(system_a)),
        ci_low=deltas[low_index],
        ci_high=deltas[high_index],
    )
