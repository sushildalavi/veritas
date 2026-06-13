"""CUDA DPO alignment for the citation-grounded verifier (Kaggle/Colab).

Aligns the CUDA QLoRA adapter (``checkpoints/cuda_qlora_verifier/``, see
``scripts/train_cuda_qlora.py``) with TRL's ``DPOTrainer`` using
``data/processed/preference_pairs.jsonl`` (built by
``scripts/build_preference_pairs_real.py``): each row is a
``{"prompt", "chosen", "rejected"}`` triple where ``chosen`` is a correct,
citation-valid Verdict/Explanation/Citation response and ``rejected`` is a
wrong-verdict, invalid-citation, or unsupported-explanation response.

Requires a CUDA GPU with ``bitsandbytes`` + ``trl`` installed (e.g. a
Colab/Kaggle T4 or better), and the CUDA QLoRA adapter to already exist. This
script will refuse to fabricate results: if CUDA, the required packages, or
the QLoRA adapter are unavailable, it writes
``reports/cuda_dpo_eval_FAILED.md`` and exits with a non-zero status instead
of producing fake metrics.

Reports before/after metrics (verdict accuracy, macro F1, citation valid
rate, unsupported-sentence rate, verdict consistency rate) on
``data/processed/mlx_lora/valid.jsonl``, comparing the QLoRA adapter (before
DPO) to the DPO-aligned adapter (after).

Outputs (on success):
  checkpoints/cuda_dpo_verifier/   (DPO-aligned LoRA adapter files)
  reports/cuda_dpo_eval.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.cuda_verifier_eval import check_cuda_environment, evaluate_adapter, load_examples
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl

METRIC_KEYS = (
    "verdict_accuracy",
    "macro_f1",
    "citation_valid_rate",
    "unsupported_sentence_rate",
    "verdict_consistency_rate",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align the CUDA QLoRA verifier adapter with DPO.")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--qlora-adapter-path", default="checkpoints/cuda_qlora_verifier")
    parser.add_argument("--preference-pairs-file", default="data/processed/preference_pairs.jsonl")
    parser.add_argument("--eval-data-dir", default="data/processed/mlx_lora")
    parser.add_argument("--eval-file", default="valid.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/cuda_dpo_verifier")
    parser.add_argument("--report-json", default="reports/cuda_dpo_eval.json")
    parser.add_argument("--report-md", default="reports/cuda_dpo_eval.md")
    parser.add_argument("--max-train-pairs", type=int, default=0, help="0 = use all preference pairs.")
    parser.add_argument("--max-val-examples", type=int, default=0, help="0 = use all examples.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    qlora_adapter_path = Path(args.qlora_adapter_path)

    try:
        if not (qlora_adapter_path / "adapter_config.json").exists():
            raise RuntimeError(
                f"{qlora_adapter_path} does not contain a trained adapter "
                "(adapter_config.json missing). Run scripts/train_cuda_qlora.py "
                "(Phase 4) first."
            )
        check_cuda_environment(("bitsandbytes", "peft", "trl"))
    except RuntimeError as exc:
        _write_failure_report(report_md, args, exc)
        raise SystemExit("PROJECT NOT COMPLETE: CUDA QLORA ADAPTER + GPU REQUIRED FOR CUDA DPO.") from exc

    import torch  # noqa: PLC0415
    from datasets import Dataset  # type: ignore  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
    from trl import DPOConfig, DPOTrainer  # noqa: PLC0415

    eval_examples = load_examples(Path(args.eval_data_dir) / args.eval_file, limit=args.max_val_examples)
    if not eval_examples:
        raise SystemExit(f"No evaluation examples found at {Path(args.eval_data_dir) / args.eval_file}")

    pairs = read_jsonl(Path(args.preference_pairs_file))
    if args.max_train_pairs > 0:
        pairs = pairs[: args.max_train_pairs]
    if not pairs:
        raise SystemExit(f"No preference pairs found at {args.preference_pairs_file}")
    train_dataset = Dataset.from_list([{"prompt": pair["prompt"], "chosen": pair["chosen"], "rejected": pair["rejected"]} for pair in pairs])

    tokenizer = AutoTokenizer.from_pretrained(str(qlora_adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, str(qlora_adapter_path), is_trainable=True)

    print(f"Evaluating QLoRA adapter (before DPO) on {len(eval_examples)} examples")
    before_report = evaluate_adapter(model, tokenizer, eval_examples, max_new_tokens=args.max_new_tokens)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        logging_strategy="epoch",
        save_strategy="no",
        report_to=[],
        fp16=True,
        bf16=False,
        fp16_full_eval=False,
        bf16_full_eval=False,
        max_grad_norm=0.0,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Evaluating DPO-aligned adapter (after DPO) on {len(eval_examples)} examples")
    after_report = evaluate_adapter(model, tokenizer, eval_examples, max_new_tokens=args.max_new_tokens)

    report = {
        "base_model": args.base_model,
        "qlora_adapter_path": str(qlora_adapter_path),
        "dpo_adapter_path": str(output_dir),
        "preference_pairs_file": args.preference_pairs_file,
        "preference_pair_count": len(pairs),
        "eval_file": str(Path(args.eval_data_dir) / args.eval_file),
        "dpo": {"beta": args.beta, "max_length": args.max_length, "max_prompt_length": args.max_prompt_length},
        "before": before_report,
        "after": after_report,
        "delta": {key: round(after_report[key] - before_report[key], 4) for key in METRIC_KEYS},
    }
    write_report(report, report_json)
    report_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Saved CUDA DPO adapter to {output_dir}")
    print(f"cuda dpo delta: {report['delta']}")


def _write_failure_report(report_md: Path, args: argparse.Namespace, exc: Exception) -> None:
    failure_path = report_md.with_name(f"{report_md.stem}_FAILED.md")
    failure_path.write_text(
        "\n".join(
            [
                "# CUDA DPO Alignment Failed",
                "",
                f"- base_model: {args.base_model}",
                f"- qlora_adapter_path: {args.qlora_adapter_path}",
                f"- output_dir: {args.output_dir}",
                "",
                "CUDA DPO alignment could not start.",
                "",
                f"Reason: {exc}",
                "",
                "No metrics were fabricated.",
            ]
        ),
        encoding="utf-8",
    )


def _to_markdown(report: dict) -> str:
    before, after, delta = report["before"], report["after"], report["delta"]
    lines = [
        "# CUDA DPO Verifier Evaluation",
        "",
        f"- base_model: {report['base_model']}",
        f"- qlora_adapter_path (before): {report['qlora_adapter_path']}",
        f"- dpo_adapter_path (after): {report['dpo_adapter_path']}",
        f"- preference_pairs_file: {report['preference_pairs_file']} ({report['preference_pair_count']} pairs)",
        f"- eval_file: {report['eval_file']}",
        f"- sample_size: {before['sample_size']}",
        "",
        "| Metric | Before (QLoRA) | After (QLoRA + DPO) | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in METRIC_KEYS:
        lines.append(f"| {key} | {before[key]} | {after[key]} | {delta[key]:+} |")

    lines += ["", "## Per-class F1", "", "| Label | Before | After |", "| --- | ---: | ---: |"]
    for label, before_score in before["per_class_f1"].items():
        lines.append(f"| {label} | {before_score} | {after['per_class_f1'].get(label, 0.0)} |")

    if after.get("top_error_examples"):
        lines += ["", "## Top error examples (after DPO)", ""]
        for example in after["top_error_examples"]:
            lines += [
                f"- claim: {example['claim']}",
                f"  - gold: {example['gold_label']}, predicted: {example['predicted_label']}",
                f"  - response: {example['response']}",
            ]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
