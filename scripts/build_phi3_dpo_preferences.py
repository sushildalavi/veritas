"""Build Phi-3 DPO preference pairs in a dedicated output directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phi-3 DPO preferences.")
    parser.add_argument("--output-dir", default="data/dpo_preferences")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/build_preference_pairs_real.py",
            "--output-jsonl",
            str(output_dir / "preferences.jsonl"),
            "--output-json",
            "reports/preference_pair_stats_phi3.json",
            "--output-md",
            "reports/preference_pair_stats_phi3.md",
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"wrote dpo preferences to {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
