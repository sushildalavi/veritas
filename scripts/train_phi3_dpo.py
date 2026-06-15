"""Train Phi-3-mini DPO adapters or emit a blocked report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Phi-3-mini with DPO.")
    parser.add_argument("--config", default="configs/phi3_dpo.yaml")
    parser.add_argument("--blocked-report", default="reports/dpo_BLOCKED_QLORA_REQUIRED.md")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    adapter_dir = Path(cfg["adapter_dir"])
    if not adapter_dir.exists():
        Path(args.blocked_report).write_text(_blocked_markdown(cfg, "QLoRA adapter checkpoint missing."), encoding="utf-8")
        print("dpo blocked: qlora adapter missing")
        raise SystemExit(1)
    if _gpu_unavailable():
        Path(args.blocked_report).write_text(_blocked_markdown(cfg, "CUDA GPU with TRL/PEFT support required."), encoding="utf-8")
        print("dpo blocked: gpu required")
        raise SystemExit(1)
    _run_training(cfg)


def _run_training(cfg: dict) -> None:  # pragma: no cover - GPU path
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = load_dataset("json", data_files=cfg["preference_file"])["train"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto")
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
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(cfg["output_dir"])
    metrics = trainer.evaluate()
    Path(cfg["report_json"]).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path(cfg["eval_json"]).write_text(json.dumps({"status": "trained", "metrics": metrics}, indent=2, sort_keys=True), encoding="utf-8")


def _gpu_unavailable() -> bool:
    try:
        import torch

        return not bool(torch.cuda.is_available())
    except Exception:
        return True


def _blocked_markdown(cfg: dict, reason: str) -> str:
    return "\n".join(
        [
            "# DPO Blocked",
            "",
            "Phi-3-mini DPO was not trained in this environment.",
            "",
            f"- Base model: `{cfg['base_model']}`",
            f"- Preference file: `{cfg['preference_file']}`",
            f"- Expected output dir: `{cfg['output_dir']}`",
            "",
            f"Reason: {reason}",
            "",
            "Use the Colab/Kaggle commands in `docs/phi3_gpu_training.md`.",
            "",
            "No checkpoint or training metrics were fabricated.",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    main()
