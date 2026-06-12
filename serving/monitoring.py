"""Structured logging and lightweight monitoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
from time import perf_counter

LOGGER = logging.getLogger("veritas")


@dataclass
class MetricsTracker:
    requests: int = 0
    fallbacks: int = 0
    verdicts: Counter[str] = field(default_factory=Counter)
    confidences: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, verdict: str, confidence: float, fallback_used: bool, latency_ms: float) -> None:
        self.requests += 1
        self.verdicts[verdict] += 1
        self.confidences.append(confidence)
        self.latencies_ms.append(latency_ms)
        if fallback_used:
            self.fallbacks += 1
        LOGGER.info(
            "verification",
            extra={
                "verdict": verdict,
                "confidence": confidence,
                "fallback_used": fallback_used,
                "latency_ms": latency_ms,
            },
        )

    def snapshot(self) -> dict[str, object]:
        average_confidence = sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
        average_latency_ms = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0
        fallback_rate = self.fallbacks / self.requests if self.requests else 0.0
        return {
            "requests": self.requests,
            "fallbacks": self.fallbacks,
            "fallback_rate": fallback_rate,
            "average_confidence": average_confidence,
            "average_latency_ms": average_latency_ms,
            "verdicts": dict(self.verdicts),
        }


def measure_latency() -> float:
    start = perf_counter()
    return start


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0
