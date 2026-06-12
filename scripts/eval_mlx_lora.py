"""Standalone evaluation for a trained MLX LoRA verifier adapter.

Loads an existing adapter (e.g. ``checkpoints/mlx_lora_verifier``) and runs
it over a larger sample of ``data/processed/verifier_*.jsonl`` than the
small smoke-eval baked into ``scripts/train_mlx_lora.py``.

Outputs:
  reports/mlx_lora_eval_<N>.{json,md} (or paths given via --output-json/--output-md)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.mlx_lora_eval import evaluate_adapter, to_markdown
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an MLX LoRA verifier adapter on a larger sample.")
    parser.add_argument("--base-model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="checkpoints/mlx_lora_verifier")
    parser.add_argument("--eval-file", default="data/processed/verifier_val.jsonl")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-json", default="reports/mlx_lora_eval_200.json")
    parser.add_argument("--output-md", default="reports/mlx_lora_eval_200.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint, requires mlx/mlx-lm
    args = build_parser().parse_args()

    eval_path = Path(args.eval_file)
    rows = read_jsonl(eval_path)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit(f"No evaluation examples found at {eval_path}")

    report = evaluate_adapter(
        args.base_model,
        args.adapter_path,
        rows,
        max_new_tokens=args.max_new_tokens,
    )
    report["eval_file"] = str(eval_path)

    write_report(report, Path(args.output_json))
    Path(args.output_md).write_text(to_markdown(report), encoding="utf-8")
    print(f"mlx lora eval ({len(rows)} examples): verdict_accuracy={report['verdict_accuracy']} macro_f1={report['macro_f1']}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
