"""Apple Silicon MLX LoRA fine-tuning for citation-grounded verification.

Fine-tunes an MLX-quantized Qwen2.5-Instruct model with a LoRA adapter on
``data/processed/mlx_lora/{train,valid,test}.jsonl`` (built by
``scripts/build_mlx_lora_dataset.py``), using ``mlx_lm.lora`` directly
(no CUDA, no bitsandbytes).

After training, evaluates the adapter on a small held-out sample: verdict
accuracy, macro F1 (if all verdicts are parseable), citation validity rate,
unsupported-sentence rate, sample size, and per-example latency.

Outputs:
  checkpoints/mlx_lora_verifier/   (LoRA adapter files)
  reports/mlx_lora_eval.{json,md}
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import yaml

from evaluation.mlx_lora_eval import evaluate_adapter, to_markdown
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune an MLX Qwen2.5 model with LoRA for fact verification.")
    parser.add_argument("--config", default="configs/mlx_lora_qwen05b.yaml")
    parser.add_argument("--model", default=None, help="Override config base model.")
    parser.add_argument("--data", default=None, help="Override config data directory.")
    parser.add_argument("--adapter-path", default=None, help="Override config adapter output path.")
    parser.add_argument("--iters", type=int, default=None, help="Override config iters (for smoke tests).")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--eval-file", default="verifier_val.jsonl")
    parser.add_argument("--max-eval-examples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--report-json", default="reports/mlx_lora_eval.json")
    parser.add_argument("--report-md", default="reports/mlx_lora_eval.md")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only evaluate an existing adapter.")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint, requires mlx/mlx-lm
    args = build_parser().parse_args()

    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser as build_lora_parser, run as run_lora  # noqa: PLC0415

    config_path = Path(args.config)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if args.iters is not None:
        config["iters"] = args.iters
    if args.model is not None:
        config["model"] = args.model
    if args.data is not None:
        config["data"] = args.data
    if args.adapter_path is not None:
        config["adapter_path"] = args.adapter_path

    lora_args = vars(build_lora_parser().parse_args([]))
    for key, value in config.items():
        lora_args[key] = value
    for key, value in CONFIG_DEFAULTS.items():
        if lora_args.get(key) is None:
            lora_args[key] = value
    lora_args["train"] = True
    lora_args["test"] = False

    adapter_path = Path(lora_args["adapter_path"])
    base_model = lora_args["model"]

    if not args.skip_train:
        adapter_path.mkdir(parents=True, exist_ok=True)
        print(f"Training LoRA adapter for {base_model} -> {adapter_path}")
        run_lora(types.SimpleNamespace(**lora_args))

    print("Loading fine-tuned model for evaluation")
    rows = read_jsonl(Path(args.data_dir) / args.eval_file)
    if args.max_eval_examples > 0:
        rows = rows[: args.max_eval_examples]
    if not rows:
        raise SystemExit(f"No evaluation examples found at {Path(args.data_dir) / args.eval_file}")

    report = evaluate_adapter(base_model, str(adapter_path), rows, max_new_tokens=args.max_new_tokens)
    report["fine_tune_type"] = lora_args.get("fine_tune_type")
    report["lora_iters"] = lora_args.get("iters")
    report["train_data"] = lora_args.get("data")
    report["eval_file"] = str(Path(args.data_dir) / args.eval_file)

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(to_markdown(report), encoding="utf-8")
    print(f"mlx lora eval: {report}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
