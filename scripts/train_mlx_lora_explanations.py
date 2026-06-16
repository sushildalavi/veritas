"""Train a Mac-local MLX LoRA adapter for grounded explanations."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl

SYSTEM_PROMPT = (
    "You are a fact-verification assistant. Use only the provided evidence. "
    "Return concise, grounded explanations with citations."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MLX LoRA explanations.")
    parser.add_argument("--config", default="configs/mlx_lora_explanations.yaml")
    parser.add_argument("--report-json", default="reports/mlx_lora_training_metrics.json")
    parser.add_argument("--report-md", default="reports/mlx_lora_training_metrics.md")
    parser.add_argument("--before-after-md", default="reports/mlx_lora_before_after_examples.md")
    parser.add_argument("--skip-train", action="store_true")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    config = _load_config(Path(args.config))
    result = {
        "status": "skipped",
        "reason": "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_path": str(Path(args.config)),
        "config": config,
        "base_model": config["base_model"],
        "dataset_path": config["dataset_dir"],
        "adapter_path": config["adapter_path"],
        "before_after_examples": [],
    }

    if args.skip_train:
        result["reason"] = "training skipped by flag"
        _write_skip_reports(result, Path(args.report_json), Path(args.report_md), Path(args.before_after_md), config)
        return

    if _training_dependencies_missing():
        result["reason"] = "mlx-lm training dependencies are unavailable in this environment"
        _write_skip_reports(result, Path(args.report_json), Path(args.report_md), Path(args.before_after_md), config)
        return

    try:
        result.update(_run_training(config))
    except Exception as exc:  # pragma: no cover - environment dependent
        result["reason"] = f"training failed: {exc}"
        _write_skip_reports(result, Path(args.report_json), Path(args.report_md), Path(args.before_after_md), config)
        return

    write_report(result, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(result), encoding="utf-8")
    Path(args.before_after_md).write_text(_before_after_markdown(result["before_after_examples"]), encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(payload, dict):
        raise ValueError("training config must be a mapping")
    defaults = {
        "base_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "dataset_dir": "data/explanations",
        "output_dir": "adapters/mlx_qwen_veritas_lora",
        "adapter_path": "adapters/mlx_qwen_veritas_lora",
        "iters": 80,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "num_layers": 8,
        "val_batches": 10,
        "steps_per_report": 20,
        "steps_per_eval": 20,
        "save_every": 40,
        "max_seq_length": 768,
        "grad_checkpoint": True,
        "seed": 42,
        "system_prompt": SYSTEM_PROMPT,
        "train_file": "sft_train.jsonl",
        "val_file": "sft_val.jsonl",
        "test_file": "sft_test.jsonl",
        "max_examples": 256,
    }
    defaults.update(payload)
    return defaults


def _training_dependencies_missing() -> bool:
    try:
        import mlx_lm  # noqa: F401
        import mlx  # noqa: F401
        return False
    except Exception:
        return True


def _run_training(config: dict[str, Any]) -> dict[str, Any]:
    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser as build_lora_parser, run as run_lora  # noqa: PLC0415

    data_dir = Path(config["dataset_dir"])
    train_rows = read_jsonl(data_dir / config["train_file"])
    val_rows = read_jsonl(data_dir / config["val_file"])
    test_rows = read_jsonl(data_dir / config["test_file"])
    if not train_rows or not val_rows:
        raise SystemExit("Grounded explanation SFT data is missing.")

    with tempfile.TemporaryDirectory(prefix="mlx_lora_explanations_") as tmp_dir:
        temp_data_dir = Path(tmp_dir) / "chat_data"
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        _write_chat_split(temp_data_dir / "train.jsonl", train_rows, config["system_prompt"], config.get("max_examples"))
        _write_chat_split(temp_data_dir / "valid.jsonl", val_rows, config["system_prompt"], config.get("max_examples"))
        _write_chat_split(temp_data_dir / "test.jsonl", test_rows, config["system_prompt"], config.get("max_examples"))

        lora_args = vars(build_lora_parser().parse_args([]))
        for key, value in config.items():
            if key in lora_args:
                lora_args[key] = value
        for key, value in CONFIG_DEFAULTS.items():
            if lora_args.get(key) is None:
                lora_args[key] = value
        lora_args["data"] = str(temp_data_dir)
        lora_args["train"] = True
        lora_args["test"] = False
        lora_args["adapter_path"] = config["adapter_path"]
        Path(config["adapter_path"]).mkdir(parents=True, exist_ok=True)
        run_lora(_namespace(lora_args))

    before_after = _collect_before_after_examples(config, train_rows[:5])
    return {
        "status": "trained",
        "reason": "",
        "trained_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "test_examples": len(test_rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "base_model": config["base_model"],
        "dataset_path": str(Path(config["dataset_dir"]).resolve()),
        "adapter_path": str(Path(config["adapter_path"]).resolve()),
        "config": config,
        "before_after_examples": before_after,
    }


def _write_chat_split(path: Path, rows: list[dict[str, Any]], system_prompt: str, max_examples: int | None) -> None:
    examples = rows[:max_examples] if max_examples else rows
    with path.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": str(row["prompt"])},
                            {"role": "assistant", "content": str(row["completion"])},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _collect_before_after_examples(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from mlx_lm import generate, load  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        return [{"status": "skipped", "reason": f"mlx-lm unavailable: {exc}"}]

    samples: list[dict[str, Any]] = []
    try:
        base_model, base_tokenizer = load(config["base_model"])
    except Exception as exc:  # pragma: no cover - environment dependent
        return [{"status": "skipped", "reason": f"base model load failed: {exc}"}]

    adapter_model = None
    adapter_tokenizer = None
    adapter_error = ""
    try:
        adapter_model, adapter_tokenizer = load(config["base_model"], adapter_path=str(config["adapter_path"]))
    except Exception as exc:  # pragma: no cover - environment dependent
        adapter_error = str(exc)

    for row in rows:
        prompt = str(row["prompt"])
        sample = {
            "claim_id": row.get("claim_id"),
            "prompt": prompt,
            "gold_completion": row.get("completion"),
            "base_output": str(generate(base_model, base_tokenizer, prompt, max_tokens=160)),
        }
        if adapter_model is not None and adapter_tokenizer is not None:
            try:
                sample["adapter_output"] = str(generate(adapter_model, adapter_tokenizer, prompt, max_tokens=160))
            except Exception as exc:  # pragma: no cover - environment dependent
                adapter_error = adapter_error or str(exc)
                sample["adapter_output"] = None
                sample["adapter_output_error"] = adapter_error
                adapter_model = None
                adapter_tokenizer = None
        else:
            sample["adapter_output"] = None
            sample["adapter_output_error"] = adapter_error
        samples.append(sample)
    return samples


def _write_skip_reports(result: dict[str, Any], report_json: Path, report_md: Path, before_after_md: Path, config: dict[str, Any]) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    result["config"] = config
    write_report(result, report_json)
    report_md.write_text(_to_markdown(result), encoding="utf-8")
    before_after_md.write_text(_before_after_markdown(result.get("before_after_examples", []), skipped_reason=result["reason"]), encoding="utf-8")


def _before_after_markdown(examples: list[dict[str, Any]], skipped_reason: str | None = None) -> str:
    lines = [
        "# MLX LoRA Before/After Examples",
        "",
    ]
    if skipped_reason:
        lines += [
            f"Training did not complete: {skipped_reason}",
            "",
        ]
    for example in examples:
        if example.get("status") == "skipped":
            lines += [f"- {example['reason']}", ""]
            continue
        lines += [
            f"## {example['claim_id']}",
            "",
            "### Prompt",
            "",
            "```text",
            example["prompt"],
            "```",
            "",
            "### Gold Completion",
            "",
            "```text",
            str(example["gold_completion"]),
            "```",
            "",
            "### Base Output",
            "",
            "```text",
            str(example["base_output"]),
            "```",
            "",
            "### Adapter Output",
            "",
            "```text",
            str(example.get("adapter_output") or example.get("adapter_output_error") or ""),
            "```",
            "",
        ]
    return "\n".join(lines)


def _to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# MLX LoRA Training Metrics",
        "",
        f"- status: {result['status']}",
        f"- reason: {result.get('reason', '')}",
        f"- timestamp_utc: {result['timestamp_utc']}",
        f"- git_commit: {result['git_commit']}",
        f"- base_model: {result['base_model']}",
        f"- dataset_path: {result['dataset_path']}",
        f"- adapter_path: {result['adapter_path']}",
        "",
    ]
    if result["status"] == "trained":
        lines += [
            f"- trained_examples: {result['trained_examples']}",
            f"- validation_examples: {result['validation_examples']}",
            f"- test_examples: {result['test_examples']}",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(result["config"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    return "\n".join(lines)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _namespace(payload: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(**payload)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
