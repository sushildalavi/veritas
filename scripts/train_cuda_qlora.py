"""CUDA QLoRA fine-tuning for the citation-grounded verifier (Kaggle/Colab).

Fine-tunes ``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` with 4-bit quantization
(bitsandbytes) and a LoRA adapter (peft) on the same
Verdict/Explanation/Citation task as the MLX LoRA adapter
(``checkpoints/mlx_lora_verifier/``), using
``data/processed/mlx_lora/{train,valid}.jsonl`` (built by
``scripts/build_mlx_lora_dataset.py``).

Requires a CUDA GPU with ``bitsandbytes`` installed (e.g. a Colab/Kaggle T4 or
better). This script will refuse to fabricate results: if CUDA or
bitsandbytes are unavailable it writes ``reports/cuda_qlora_eval_FAILED.md``
and exits with a non-zero status instead of producing fake metrics.

``--export-sft-only`` writes the flattened ``{"prompt", "completion"}`` SFT
dataset (``data/processed/sft_train.jsonl`` / ``sft_val.jsonl``) without
requiring a GPU, for inspection or reuse outside this script.

Outputs (on success):
  data/processed/sft_train.jsonl / sft_val.jsonl
  checkpoints/cuda_qlora_verifier/   (LoRA adapter files)
  reports/cuda_qlora_eval.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from evaluation.cuda_verifier_eval import Example, check_cuda_environment, evaluate_adapter, load_examples, write_sft_jsonl
from evaluation.mlx_lora_eval import to_markdown
from evaluation.reporting import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with CUDA QLoRA for fact verification.")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data-dir", default="data/processed/mlx_lora")
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--val-file", default="valid.jsonl")
    parser.add_argument("--sft-train-file", default="data/processed/sft_train.jsonl")
    parser.add_argument("--sft-val-file", default="data/processed/sft_val.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/cuda_qlora_verifier")
    parser.add_argument("--report-json", default="reports/cuda_qlora_eval.json")
    parser.add_argument("--report-md", default="reports/cuda_qlora_eval.md")
    parser.add_argument("--max-train-examples", type=int, default=0, help="0 = use all examples.")
    parser.add_argument("--max-val-examples", type=int, default=0, help="0 = use all examples.")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--export-sft-only",
        action="store_true",
        help="Only write the flattened SFT dataset (no GPU required) and exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)

    train_examples = load_examples(data_dir / args.train_file, limit=args.max_train_examples)
    val_examples = load_examples(data_dir / args.val_file, limit=args.max_val_examples)
    if not train_examples:
        raise SystemExit(f"No training examples found at {data_dir / args.train_file}")

    write_sft_jsonl(Path(args.sft_train_file), train_examples)
    write_sft_jsonl(Path(args.sft_val_file), val_examples)
    print(f"wrote {len(train_examples)} train / {len(val_examples)} val examples to {args.sft_train_file} / {args.sft_val_file}")

    if args.export_sft_only:
        return

    output_dir = Path(args.output_dir)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        check_cuda_environment(("bitsandbytes",))
    except RuntimeError as exc:
        failure_path = report_md.with_name(f"{report_md.stem}_FAILED.md")
        failure_path.write_text(
            "\n".join(
                [
                    "# CUDA QLoRA Training Failed",
                    "",
                    f"- base_model: {args.base_model}",
                    f"- output_dir: {output_dir}",
                    "",
                    "CUDA QLoRA training could not start.",
                    "",
                    f"Reason: {exc}",
                    "",
                    "No metrics were fabricated.",
                ]
            ),
            encoding="utf-8",
        )
        raise SystemExit("PROJECT NOT COMPLETE: GPU REQUIRED FOR CUDA QLORA.") from exc

    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = _build_dataset(train_examples, tokenizer, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        logging_strategy="epoch",
        save_strategy="no",
        report_to=[],
        fp16=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    started = perf_counter()
    trainer.train()
    training_runtime = round(perf_counter() - started, 3)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    report = evaluate_adapter(model, tokenizer, val_examples, max_new_tokens=args.max_new_tokens)
    report["base_model"] = args.base_model
    report["adapter_path"] = str(output_dir)
    report["fine_tune_type"] = "qlora"
    report["lora"] = {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout}
    report["train_data"] = str(data_dir / args.train_file)
    report["eval_file"] = str(data_dir / args.val_file)
    report["train_example_count"] = len(train_examples)
    report["training_runtime_seconds"] = training_runtime

    write_report(report, report_json)
    report_md.write_text(to_markdown(report, title="CUDA QLoRA Verifier Evaluation"), encoding="utf-8")
    print(f"Saved CUDA QLoRA adapter to {output_dir}")
    print(f"cuda qlora eval: verdict_accuracy={report['verdict_accuracy']} macro_f1={report['macro_f1']}")


def _build_dataset(examples: list[Example], tokenizer, max_seq_length: int):  # noqa: ANN001
    from datasets import Dataset  # type: ignore

    texts = [f"{example.prompt}\n{example.completion}{tokenizer.eos_token}" for example in examples]

    def _tokenize(batch):  # noqa: ANN001
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_seq_length, padding="max_length")
        encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
        return encoded

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return tokenized


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
