import json

from serving.vllm_client import VllmExplanationGenerator, _extract_message_text


def test_extract_message_text_prefers_chat_content() -> None:
    payload = {"choices": [{"message": {"content": "{\"explanation\":\"ok\"}"}}]}

    assert _extract_message_text(payload) == '{"explanation":"ok"}'


def test_vllm_generator_posts_openai_compatible_payload(monkeypatch) -> None:
    seen = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "done"}}]}).encode("utf-8")

    def fake_urlopen(req, timeout):  # noqa: ANN001
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("serving.vllm_client.request.urlopen", fake_urlopen)
    generator = VllmExplanationGenerator(
        base_url="http://127.0.0.1:8001",
        model="Qwen/Test",
        api_key="secret",
        timeout_seconds=12.0,
    )

    result = generator("Explain this claim.")

    assert result == "done"
    assert seen["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert seen["body"]["model"] == "Qwen/Test"
    assert seen["body"]["messages"][0]["content"] == "Explain this claim."
    assert seen["timeout"] == 12.0
