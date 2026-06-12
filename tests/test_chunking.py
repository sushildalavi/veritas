import pytest

from data.chunking import ChunkConfig, chunk_text


def test_chunk_text_overlapping_windows() -> None:
    text = "one two three four five six seven"
    chunks = chunk_text(text, ChunkConfig(chunk_size=3, overlap=1))

    assert chunks == ["one two three", "three four five", "five six seven"]


def test_chunk_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        ChunkConfig(chunk_size=4, overlap=4)
