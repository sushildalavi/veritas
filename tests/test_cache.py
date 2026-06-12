from serving.cache import ResponseCache


def test_response_cache_expires_entries() -> None:
    cache = ResponseCache(ttl_seconds=0)

    cache.set("key", {"value": 1})

    assert cache.get("key") is None
    assert cache.size() == 0
