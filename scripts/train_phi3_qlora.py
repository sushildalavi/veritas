"""Train Phi-3-mini QLoRA adapters or emit a skipped report."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Phi-3-mini with QLoRA.")
    parser.add_argument("--config", default="configs/phi3_qlora.yaml")
    parser.add_argument("--report-json", default="reports/phi3_qlora_skipped_or_training_metrics.json")
    parser.add_argument("--report-md", default="reports/phi3_qlora_skipped_or_training_metrics.md")
    parser.add_argument("--before-after-md", default="reports/phi3_qlora_before_after_examples.md")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write a non-training readiness report.")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    cfg = _load_config(Path(args.config))
    if args.dry_run:
        payload = _dry_run_payload(cfg)
        _write_reports(payload, Path(args.report_json), Path(args.report_md), Path(args.before_after_md))
        return
    if not _cuda_training_ready():
        payload = _skipped_payload(cfg, reason=_skip_reason())
        _write_reports(payload, Path(args.report_json), Path(args.report_md), Path(args.before_after_md))
        raise SystemExit(1)
    payload = _run_training(cfg)
    _write_reports(payload, Path(args.report_json), Path(args.report_md), Path(args.before_after_md))


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(payload, dict):
        raise ValueError("config must be a mapping")
    defaults = {
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "train_file": "data/explanations/sft_train.jsonl",
        "eval_file": "data/explanations/sft_val.jsonl",
        "output_dir": "adapters/phi3_veritas_qlora",
        "max_seq_length": 2048,
        "learning_rate": 2.0e-4,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    defaults.update(payload)
    return defaults


def _cuda_training_ready() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import peft  # noqa: F401
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _skip_reason() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "CUDA is unavailable on this machine."
    except Exception as exc:
        return f"torch check failed: {exc}"
    try:
        import peft  # noqa: F401
    except Exception as exc:
        return f"peft is unavailable: {exc}"
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        return f"bitsandbytes is unavailable: {exc}"
    return "unknown environment limitation"


def _run_training(cfg: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - GPU path
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    dataset = load_dataset("json", data_files={"train": cfg["train_file"], "eval": cfg["eval_file"]})
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg["target_modules"]),
    )
    model = get_peft_model(model, peft_config)

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = f"{example['prompt']}\n\n{example['completion']}"
        tokenized = tokenizer(text, truncation=True, max_length=cfg["max_seq_length"])
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    train_dataset = dataset["train"].map(tokenize, remove_columns=dataset["train"].column_names)
    eval_dataset = dataset["eval"].map(tokenize, remove_columns=dataset["eval"].column_names)
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        fp16=False,
        bf16=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    metrics = trainer.evaluate()
    samples = _sample_generations(cfg, max_examples=5)
    return {
        "status": "trained",
        "reason": "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "base_model": cfg["base_model"],
        "dataset_path": str(Path(cfg["train_file"]).resolve().parent),
        "train_file": cfg["train_file"],
        "eval_file": cfg["eval_file"],
        "adapter_path": str(Path(cfg["output_dir"]).resolve()),
        "config": cfg,
        "train_metrics": metrics,
        "before_after_examples": samples,
    }


def _dry_run_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "reason": "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "base_model": cfg["base_model"],
        "train_file": cfg["train_file"],
        "eval_file": cfg["eval_file"],
        "adapter_path": cfg["output_dir"],
        "config": cfg,
        "train_metrics": {},
        "before_after_examples": [],
        "preflight_checks": _preflight_checks(cfg, require_adapter=False),
        "colab_commands": _colab_commands("train_phi3_qlora.py", "configs/phi3_qlora.yaml"),
        "kaggle_commands": _colab_commands("train_phi3_qlora.py", "configs/phi3_qlora.yaml"),
    }


def _sample_generations(cfg: dict[str, Any], *, max_examples: int = 5) -> list[dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = read_jsonl(Path(cfg["eval_file"]))[:max_examples]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    base_model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant_config, device_map="auto")
    adapter_model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant_config, device_map="auto")
    try:
        from peft import PeftModel

        adapter_model = PeftModel.from_pretrained(adapter_model, cfg["output_dir"])
    except Exception:
        pass

    samples: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["prompt"])
        base_output = _generate_text(base_model, tokenizer, prompt)
        adapter_output = _generate_text(adapter_model, tokenizer, prompt)
        samples.append(
            {
                "claim_id": row.get("claim_id"),
                "prompt": prompt,
                "gold_completion": row.get("completion"),
                "base_output": base_output,
                "adapter_output": adapter_output,
            }
        )
    return samples


def _generate_text(model: Any, tokenizer: Any, prompt: str) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=160)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _skipped_payload(cfg: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "base_model": cfg["base_model"],
        "train_file": cfg["train_file"],
        "eval_file": cfg["eval_file"],
        "adapter_path": cfg["output_dir"],
        "config": cfg,
        "train_metrics": {},
        "before_after_examples": [],
        "preflight_checks": _preflight_checks(cfg, require_adapter=False),
        "colab_commands": _colab_commands("train_phi3_qlora.py", "configs/phi3_qlora.yaml"),
        "kaggle_commands": _colab_commands("train_phi3_qlora.py", "configs/phi3_qlora.yaml"),
    }


def _preflight_checks(cfg: dict[str, Any], *, require_adapter: bool) -> list[dict[str, Any]]:
    checks = [
        {"name": "train_file_exists", "path": cfg["train_file"], "exists": Path(cfg["train_file"]).exists()},
        {"name": "eval_file_exists", "path": cfg["eval_file"], "exists": Path(cfg["eval_file"]).exists()},
        {"name": "output_parent_exists", "path": str(Path(cfg["output_dir"]).parent), "exists": Path(cfg["output_dir"]).parent.exists()},
        {"name": "base_model", "path": cfg["base_model"], "exists": True},
    ]
    if require_adapter:
        checks.append({"name": "adapter_exists", "path": cfg["output_dir"], "exists": Path(cfg["output_dir"]).exists()})
    return checks


def _colab_commands(script_name: str, config_name: str) -> list[str]:
    return [
        "pip install transformers datasets peft trl accelerate bitsandbytes",
        f"python3 scripts/{script_name} --config {config_name}",
    ]


def _write_reports(payload: dict[str, Any], report_json: Path, report_md: Path, before_after_md: Path) -> None:
    write_report(payload, report_json)
    report_md.write_text(_to_markdown(payload), encoding="utf-8")
    before_after_md.write_text(_before_after_markdown(payload.get("before_after_examples", []), payload["status"], payload.get("reason", "")), encoding="utf-8")


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phi-3 QLoRA Training Report",
        "",
        f"- status: {payload['status']}",
        f"- reason: {payload.get('reason', '')}",
        f"- timestamp_utc: {payload['timestamp_utc']}",
        f"- git_commit: {payload['git_commit']}",
        f"- base_model: {payload['base_model']}",
        f"- train_file: {payload['train_file']}",
        f"- eval_file: {payload['eval_file']}",
        f"- adapter_path: {payload['adapter_path']}",
        "",
    ]
    if payload.get("preflight_checks"):
        lines += [
            "## Preflight Checks",
            "",
            "| Check | Path | Exists |",
            "| --- | --- | ---: |",
        ]
        for check in payload["preflight_checks"]:
            lines.append(f"| {check['name']} | {check['path']} | {str(check['exists']).lower()} |")
        lines.append("")
    if payload["status"] == "trained":
        lines += [
            "## Training Metrics",
            "",
            "```json",
            json.dumps(payload["train_metrics"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    elif payload["status"] == "dry_run":
        lines += [
            "## Colab / Kaggle Commands",
            "",
            "```bash",
            "\n".join(payload["colab_commands"]),
            "```",
            "",
            "## Run Notes",
            "",
            "- This is a readiness check only.",
            "- No training was executed.",
            "- The report confirms local file paths and command syntax.",
            "",
        ]
    else:
        lines += [
            "## Environment",
            "",
            f"- reason: {payload['reason']}",
            "",
            "## Colab / Kaggle Commands",
            "",
            "```bash",
            "\n".join(payload["colab_commands"]),
            "```",
            "",
        ]
    return "\n".join(lines)


def _before_after_markdown(examples: list[dict[str, Any]], status: str, reason: str) -> str:
    lines = [
        "# Phi-3 QLoRA Before/After Examples",
        "",
    ]
    if status != "trained":
        lines += [f"Training did not complete: {reason}", ""]
        return "\n".join(lines)
    for example in examples:
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
            str(example["adapter_output"]),
            "```",
            "",
        ]
    return "\n".join(lines)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":  # pragma: no cover
    main()
