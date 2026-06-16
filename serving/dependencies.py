"""FastAPI dependency providers for the Veritas API."""

from __future__ import annotations

from functools import lru_cache

from core.config import ProjectSettings, load_project_settings
from serving.cache import ResponseCache
from serving.model_loader import VerificationPipeline, load_pipeline
from serving.monitoring import MetricsTracker


@lru_cache(maxsize=1)
def get_settings() -> ProjectSettings:
    return load_project_settings()


@lru_cache(maxsize=1)
def get_pipeline() -> VerificationPipeline:
    return load_pipeline(settings=get_settings())


@lru_cache(maxsize=1)
def get_cache() -> ResponseCache:
    return ResponseCache(ttl_seconds=get_settings().cache_ttl_seconds)


@lru_cache(maxsize=1)
def get_metrics() -> MetricsTracker:
    return MetricsTracker()
