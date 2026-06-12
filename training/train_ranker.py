"""Learned ranker training script placeholder."""

from __future__ import annotations

import argparse


def main() -> None:  # pragma: no cover - script entrypoint
    parser = argparse.ArgumentParser(description="Run the learned ranker training pipeline.")
    parser.add_argument("--config", default="configs/ranking.yaml")
    args = parser.parse_args()
    print(f"Loaded ranker config from {args.config}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
