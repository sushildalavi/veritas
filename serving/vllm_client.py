"""OpenAI-compatible vLLM client for explanation generation."""

from __future__ import annotations

import json
from urllib import error, request


class VllmExplanationGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exc: Exception | None = None
        for _ in range(max(1, self.max_retries + 1)):
            req = request.Request(
                url=f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:  # noqa: S310
                    body = json.loads(response.read().decode("utf-8"))
                return _extract_message_text(body)
            except (error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc

        return _fallback_response(last_exc)


def _extract_message_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
        return str(content)
    text = first.get("text", "")
    return str(text)


def _fallback_response(exc: Exception | None) -> str:
    """JSON fallback returned when all vLLM request attempts fail.

    Shaped like the expected ``{"explanation": ..., "citations": [...]}``
    response so downstream parsing (``rag.explanation_generator``) degrades
    cleanly instead of raising.
    """

    reason = str(exc) if exc is not None else "unknown error"
    return json.dumps({"explanation": f"Explanation unavailable: vLLM request failed ({reason}).", "citations": []})
