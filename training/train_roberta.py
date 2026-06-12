"""RoBERTa baseline training script."""

from __future__ import annotations

import argparse

from training.config import load_yaml


def load_config(path: str) -> dict[str, object]:
    return load_yaml(path)


def main() -> None:  # pragma: no cover - script entrypoint
    parser = argparse.ArgumentParser(description="Run the RoBERTa baseline training pipeline.")
    parser.add_argument("--config", default="configs/deberta.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    model_name = config.get("model", {}).get("name", "unknown model") if isinstance(config.get("model"), dict) else "unknown model"
    print(f"Loaded RoBERTa config for {model_name}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
