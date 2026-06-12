"""Convenience wrapper for the reflection loop."""

from __future__ import annotations

from .reflection import ReflectionLoop, ReflectionOutcome


def run_reflection_graph(loop: ReflectionLoop, claim: str, *, top_k: int = 5) -> ReflectionOutcome:
    return loop.run(claim, top_k=top_k)
