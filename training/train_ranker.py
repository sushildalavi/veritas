"""Learned ranker training script."""

from __future__ import annotations

import argparse

from training.config import load_yaml


def load_config(path: str) -> dict[str, object]:
    return load_yaml(path)


def main() -> None:  # pragma: no cover - script entrypoint
    parser = argparse.ArgumentParser(description="Run the learned ranker training pipeline.")
    parser.add_argument("--config", default="configs/ranking.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    model_backend = config.get("model", {}).get("backend", "auto") if isinstance(config.get("model"), dict) else "auto"
    print(f"Loaded ranker config with backend {model_backend}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
