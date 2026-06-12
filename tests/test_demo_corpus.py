from data.demo_corpus import DEFAULT_DEMO_PASSAGES


def test_demo_corpus_is_non_empty() -> None:
    assert len(DEFAULT_DEMO_PASSAGES) >= 3
