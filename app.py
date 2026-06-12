"""Hugging Face Spaces entrypoint for the Veritas demo."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from ui.app import build_demo
except ModuleNotFoundError:  # pragma: no cover - local test environment fallback
    build_demo = None  # type: ignore[assignment]


@dataclass
class _FallbackApp:
    message: str = "gradio is unavailable in this environment"

    def launch(self, *args, **kwargs):  # pragma: no cover - fallback only
        print(self.message)
        return self


app = build_demo() if build_demo is not None else _FallbackApp()


if __name__ == "__main__":  # pragma: no cover - Spaces/local entrypoint
    app.launch()
