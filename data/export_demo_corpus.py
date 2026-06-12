"""Export the built-in demo corpus as JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from .demo_corpus import DEFAULT_DEMO_PASSAGES


def export_demo_corpus(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, passage in enumerate(DEFAULT_DEMO_PASSAGES):
            handle.write(
                json.dumps(
                    {
                        "doc_id": str(index),
                        "text": passage,
                        "title": f"demo-{index}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path
