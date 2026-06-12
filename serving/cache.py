"""Simple in-process response cache."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass
class CacheEntry:
    value: object
    expires_at: float


class ResponseCache:
    def __init__(self, ttl_seconds: int = 60) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at < monotonic():
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: object) -> None:
        self._store[key] = CacheEntry(value=value, expires_at=monotonic() + self.ttl_seconds)

    def size(self) -> int:
        self._purge_expired()
        return len(self._store)

    def _purge_expired(self) -> None:
        now = monotonic()
        for key, entry in list(self._store.items()):
            if entry.expires_at < now:
                self._store.pop(key, None)
