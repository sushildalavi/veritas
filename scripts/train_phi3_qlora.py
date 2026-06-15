"""Train Phi-3-mini QLoRA adapters or emit a blocked report on CPU-only hosts."""

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
    parser = argparse.ArgumentParser(description="Train Phi-3-mini with QLoRA.")
    parser.add_argument("--config", default="configs/phi3_qlora.yaml")
    parser.add_argument("--blocked-report", default="reports/qlora_BLOCKED_GPU_REQUIRED.md")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    if _gpu_unavailable():
        Path(args.blocked_report).write_text(_blocked_markdown(cfg), encoding="utf-8")
        print("qlora blocked: gpu required")
        raise SystemExit(1)
    _run_training(cfg)


def _run_training(cfg: dict) -> None:  # pragma: no cover - GPU path
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    dataset = load_dataset("json", data_files={"train": cfg["train_file"], "eval": cfg["eval_file"]})
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant_config, device_map="auto")
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg["target_modules"]),
    )
    model = get_peft_model(model, peft_config)

    def tokenize(example):  # noqa: ANN001
        text = "\n".join(message["content"] for message in example["messages"])
        tokenized = tokenizer(text, truncation=True, max_length=cfg["max_seq_length"])
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    train_dataset = dataset["train"].map(tokenize)
    eval_dataset = dataset["eval"].map(tokenize)
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
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    metrics = trainer.evaluate()
    Path(cfg["report_json"]).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path(cfg["before_after_md"]).write_text(
        "# QLoRA Before/After Examples\n\nTraining completed. Add sampled generations here.\n",
        encoding="utf-8",
    )


def _gpu_unavailable() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return True
        import bitsandbytes  # noqa: F401
        import peft  # noqa: F401
        return False
    except Exception:
        return True


def _blocked_markdown(cfg: dict) -> str:
    return "\n".join(
        [
            "# QLoRA Blocked",
            "",
            "Phi-3-mini QLoRA was not trained in this environment.",
            "",
            f"- Base model: `{cfg['base_model']}`",
            f"- Train file: `{cfg['train_file']}`",
            f"- Expected output dir: `{cfg['output_dir']}`",
            "",
            "Reason: CUDA GPU with bitsandbytes/PEFT support is required.",
            "",
            "Use the Colab/Kaggle commands in `docs/phi3_gpu_training.md`.",
            "",
            "No checkpoint or training metrics were fabricated.",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    main()
