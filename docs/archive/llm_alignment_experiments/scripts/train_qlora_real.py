"""Real QLoRA fine-tuning for citation-grounded explanation generation.

Fine-tunes ``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` with 4-bit quantization
(bitsandbytes) and a LoRA adapter (peft) on claim/evidence/verdict ->
citation-grounded explanation pairs derived from
``data/processed/verifier_{train,val}.jsonl``.

Requires a CUDA GPU with ``bitsandbytes`` installed (e.g. a Colab/Kaggle T4 or
better). This script will refuse to fabricate results: if CUDA or
bitsandbytes are unavailable it writes ``reports/qlora_eval_FAILED.md`` and
exits with a non-zero status instead of producing fake metrics.

Outputs (on success):
  checkpoints/qlora_tinyllama/   (LoRA adapter files)
  reports/qlora_eval.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from rag import build_context, check_citations, generate_template_explanation
from rag.prompt_templates import EXPLANATION_PROMPT
from models.deberta_verifier import VerificationResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with QLoRA for explanation generation.")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="checkpoints/qlora_tinyllama")
    parser.add_argument("--report-json", default="reports/qlora_eval.json")
    parser.add_argument("--report-md", default="reports/qlora_eval.md")
    parser.add_argument("--max-train-examples", type=int, default=2000)
    parser.add_argument("--max-val-examples", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint, requires CUDA + bitsandbytes
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        _check_gpu_environment()
    except RuntimeError as exc:
        failure_path = report_md.with_name(f"{report_md.stem}_FAILED.md")
        failure_path.write_text(
            "\n".join(
                [
                    "# QLoRA Training Failed",
                    "",
                    f"- base_model: {args.base_model}",
                    f"- output_dir: {output_dir}",
                    "",
                    "QLoRA training could not start.",
                    "",
                    f"Reason: {exc}",
                    "",
                    "No metrics were fabricated.",
                ]
            ),
            encoding="utf-8",
        )
        raise SystemExit("PROJECT NOT COMPLETE: GPU REQUIRED FOR QLORA.") from exc

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
    data_dir = Path(args.data_dir)

    train_examples = _build_examples(data_dir / "verifier_train.jsonl", limit=args.max_train_examples)
    val_examples = _build_examples(data_dir / "verifier_val.jsonl", limit=args.max_val_examples)
    if not train_examples:
        raise SystemExit("No training examples found; run scripts/build_verifier_dataset.py first")

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

    eval_metrics = _evaluate_explanations(model, tokenizer, val_examples, args.max_seq_length)
    report = {
        "base_model": args.base_model,
        "adapter_path": str(output_dir),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "train_example_count": len(train_examples),
        "val_example_count": len(val_examples),
        "training_runtime_seconds": training_runtime,
        "explanation_metrics": eval_metrics,
    }
    write_report(report, report_json)
    report_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Saved QLoRA adapter to {output_dir}")
    print(f"explanation metrics: {eval_metrics}")


def _check_gpu_environment() -> None:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; QLoRA requires a CUDA GPU.")
    try:
        import bitsandbytes  # noqa: F401, PLC0415
    except ImportError as exc:
        raise RuntimeError("bitsandbytes is not installed; required for 4-bit QLoRA quantization.") from exc


class _Example:
    __slots__ = ("claim", "evidence_text", "label", "prompt", "target")

    def __init__(self, claim: str, evidence_text: str, label: str) -> None:
        self.claim = claim
        self.evidence_text = evidence_text
        self.label = label
        context = build_context(claim, [EvidenceSpan(doc_id="1", text=evidence_text)] if evidence_text else [])
        verification = VerificationResult(verdict=_display_label(label), confidence=1.0)
        template = generate_template_explanation(context, verification)
        self.prompt = EXPLANATION_PROMPT.format(
            claim=claim,
            verdict=_display_label(label),
            evidence_block=context.format_block() or "(no evidence retrieved)",
        )
        self.target = template.explanation


def _display_label(label: str) -> str:
    return "NOT ENOUGH INFO" if label == "NOT_ENOUGH_INFO" else label


def _build_examples(path: Path, *, limit: int) -> list[_Example]:
    rows = read_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    return [
        _Example(
            claim=str(row.get("claim", "")),
            evidence_text=str(row.get("evidence", "")),
            label=str(row.get("label", "NOT_ENOUGH_INFO")),
        )
        for row in rows
    ]


def _build_dataset(examples: list[_Example], tokenizer, max_seq_length: int):  # noqa: ANN001
    from datasets import Dataset  # type: ignore

    texts = [f"{example.prompt}\n{example.target}{tokenizer.eos_token}" for example in examples]

    def _tokenize(batch):  # noqa: ANN001
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_seq_length, padding="max_length")
        encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
        return encoded

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return tokenized


def _evaluate_explanations(model, tokenizer, examples: list[_Example], max_seq_length: int) -> dict[str, object]:  # noqa: ANN001
    import torch  # noqa: PLC0415

    rows = []
    model.eval()
    for example in examples:
        inputs = tokenizer(example.prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        context = build_context(example.claim, [EvidenceSpan(doc_id="1", text=example.evidence_text)] if example.evidence_text else [])
        citation_result = check_citations(text, context)
        rows.append(
            {
                "citation_valid": citation_result.valid,
                "citation_precision": citation_result.citation_precision,
                "verdict_consistency": citation_result.verdict_consistency,
            }
        )

    if not rows:
        return {"example_count": 0, "citation_valid_rate": 0.0, "mean_citation_precision": 0.0, "verdict_consistency_rate": 0.0}

    return {
        "example_count": len(rows),
        "citation_valid_rate": sum(1.0 for row in rows if row["citation_valid"]) / len(rows),
        "mean_citation_precision": sum(row["citation_precision"] for row in rows) / len(rows),
        "verdict_consistency_rate": sum(1.0 for row in rows if row["verdict_consistency"]) / len(rows),
    }


def _to_markdown(report: dict[str, object]) -> str:
    metrics = report["explanation_metrics"]
    lines = [
        "# QLoRA Explanation Fine-Tuning Evaluation",
        "",
        f"- Base model: `{report['base_model']}`",
        f"- Adapter path: `{report['adapter_path']}`",
        f"- LoRA config: {report['lora']}",
        f"- Train examples: {report['train_example_count']}",
        f"- Val examples: {report['val_example_count']}",
        f"- Training runtime (s): {report['training_runtime_seconds']}",
        "",
        "## Explanation faithfulness (validation set)",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| example_count | {metrics['example_count']} |",
        f"| citation_valid_rate | {metrics['citation_valid_rate']:.3f} |",
        f"| mean_citation_precision | {metrics['mean_citation_precision']:.3f} |",
        f"| verdict_consistency_rate | {metrics['verdict_consistency_rate']:.3f} |",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
