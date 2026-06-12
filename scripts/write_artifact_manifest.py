"""Write a manifest of generated artifacts and active model backends."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_project_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an artifact manifest for Veritas outputs.")
    parser.add_argument("--output", default="reports/artifact_manifest.json")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    settings = load_project_settings()
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "model_backends": {
            "verifier_backend": settings.verifier_backend,
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model,
            "cross_encoder_model": settings.cross_encoder_model,
            "sklearn_checkpoint": settings.sklearn_checkpoint,
            "transformer_checkpoint": settings.transformer_checkpoint,
        },
        "processed_datasets": _list_artifacts("data/processed"),
        "reports": _list_artifacts("reports"),
        "checkpoints": _list_artifacts("checkpoints"),
        "artifacts": _list_artifacts("artifacts"),
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote artifact manifest to {output}")


def _list_artifacts(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file() and p.name != ".gitkeep"):
        items.append(
            {
                "path": str(path.as_posix()),
                "size_bytes": path.stat().st_size,
            }
        )
    return items


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
