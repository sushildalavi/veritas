"""Train Phi-3-mini with DPO or emit a skipped report."""

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
    parser = argparse.ArgumentParser(description="Train Phi-3-mini with DPO.")
    parser.add_argument("--config", default="configs/phi3_dpo.yaml")
    parser.add_argument("--report-json", default="reports/phi3_dpo_skipped_or_training_metrics.json")
    parser.add_argument("--report-md", default="reports/phi3_dpo_skipped_or_training_metrics.md")
    parser.add_argument("--before-after-md", default="reports/phi3_dpo_before_after_examples.md")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    cfg = _load_config(Path(args.config))
    if not _cuda_training_ready():
        payload = _skipped_payload(cfg, reason=_skip_reason())
        _write_reports(payload, Path(args.report_json), Path(args.report_md), Path(args.before_after_md))
        raise SystemExit(1)
    if not Path(cfg["adapter_dir"]).exists():
        payload = _skipped_payload(cfg, reason=f"required QLoRA adapter is missing: {cfg['adapter_dir']}")
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
        "adapter_dir": "adapters/phi3_veritas_qlora",
        "preference_file": "data/explanations/dpo_train.jsonl",
        "eval_file": "data/explanations/dpo_val.jsonl",
        "output_dir": "adapters/phi3_veritas_dpo",
        "max_length": 2048,
        "learning_rate": 5.0e-6,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "beta": 0.1,
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
        import trl  # noqa: F401
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
    try:
        import trl  # noqa: F401
    except Exception as exc:
        return f"trl is unavailable: {exc}"
    return "unknown environment limitation"


def _run_training(cfg: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - GPU path
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    dataset = load_dataset("json", data_files={"train": cfg["preference_file"], "eval": cfg["eval_file"]})
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    policy_model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant_config, device_map="auto")
    policy_model = PeftModel.from_pretrained(policy_model, cfg["adapter_dir"], is_trainable=True)
    reference_model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant_config, device_map="auto")

    training_args = DPOConfig(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        beta=cfg["beta"],
        report_to=[],
    )
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    metrics = trainer.evaluate()
    samples = _sample_outputs(cfg, max_examples=5)
    return {
        "status": "trained",
        "reason": "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "base_model": cfg["base_model"],
        "adapter_dir": cfg["adapter_dir"],
        "preference_file": cfg["preference_file"],
        "eval_file": cfg["eval_file"],
        "output_dir": str(Path(cfg["output_dir"]).resolve()),
        "config": cfg,
        "train_metrics": metrics,
        "before_after_examples": samples,
    }


def _sample_outputs(cfg: dict[str, Any], *, max_examples: int = 5) -> list[dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = read_jsonl(Path(cfg["eval_file"]))[:max_examples]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    base_model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant_config, device_map="auto")
    policy_model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant_config, device_map="auto")
    try:
        from peft import PeftModel

        policy_model = PeftModel.from_pretrained(policy_model, cfg["output_dir"])
    except Exception:
        pass

    samples: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["prompt"])
        samples.append(
            {
                "claim_id": row.get("claim_id"),
                "prompt": prompt,
                "chosen": row.get("chosen"),
                "rejected": row.get("rejected"),
                "base_output": _generate_text(base_model, tokenizer, prompt),
                "policy_output": _generate_text(policy_model, tokenizer, prompt),
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
        "adapter_dir": cfg["adapter_dir"],
        "preference_file": cfg["preference_file"],
        "eval_file": cfg["eval_file"],
        "output_dir": cfg["output_dir"],
        "config": cfg,
        "train_metrics": {},
        "before_after_examples": [],
        "colab_commands": [
            "pip install transformers datasets peft trl accelerate bitsandbytes",
            "python3 scripts/train_phi3_qlora.py --config configs/phi3_qlora.yaml",
            "python3 scripts/train_phi3_dpo.py --config configs/phi3_dpo.yaml",
        ],
        "kaggle_commands": [
            "pip install transformers datasets peft trl accelerate bitsandbytes",
            "python3 scripts/train_phi3_qlora.py --config configs/phi3_qlora.yaml",
            "python3 scripts/train_phi3_dpo.py --config configs/phi3_dpo.yaml",
        ],
    }


def _write_reports(payload: dict[str, Any], report_json: Path, report_md: Path, before_after_md: Path) -> None:
    write_report(payload, report_json)
    report_md.write_text(_to_markdown(payload), encoding="utf-8")
    before_after_md.write_text(_before_after_markdown(payload.get("before_after_examples", []), payload["status"], payload.get("reason", "")), encoding="utf-8")


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phi-3 DPO Training Report",
        "",
        f"- status: {payload['status']}",
        f"- reason: {payload.get('reason', '')}",
        f"- timestamp_utc: {payload['timestamp_utc']}",
        f"- git_commit: {payload['git_commit']}",
        f"- base_model: {payload['base_model']}",
        f"- adapter_dir: {payload['adapter_dir']}",
        f"- preference_file: {payload['preference_file']}",
        f"- eval_file: {payload['eval_file']}",
        f"- output_dir: {payload['output_dir']}",
        "",
    ]
    if payload["status"] == "trained":
        lines += [
            "## Training Metrics",
            "",
            "```json",
            json.dumps(payload["train_metrics"], indent=2, sort_keys=True),
            "```",
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
        "# Phi-3 DPO Before/After Examples",
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
            "### Chosen",
            "",
            "```text",
            str(example["chosen"]),
            "```",
            "",
            "### Rejected",
            "",
            "```text",
            str(example["rejected"]),
            "```",
            "",
            "### Base Output",
            "",
            "```text",
            str(example["base_output"]),
            "```",
            "",
            "### Policy Output",
            "",
            "```text",
            str(example["policy_output"]),
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
