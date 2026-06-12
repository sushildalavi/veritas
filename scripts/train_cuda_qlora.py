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
import json
import re
from pathlib import Path
from time import perf_counter

from data.schemas import EvidenceSpan
from evaluation.mlx_lora_eval import (
    EXPLANATION_RE,
    LABEL_ORDER,
    UNPARSEABLE,
    VERDICT_RE,
    to_markdown,
)
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from rag import build_context, check_citations

CLAIM_RE = re.compile(r"Claim: (.*?)\nEvidence:", re.DOTALL)
EVIDENCE_RE = re.compile(r"Evidence: \[1\] (.*?)\nClassify", re.DOTALL)


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

    train_examples = _load_examples(data_dir / args.train_file, limit=args.max_train_examples)
    val_examples = _load_examples(data_dir / args.val_file, limit=args.max_val_examples)
    if not train_examples:
        raise SystemExit(f"No training examples found at {data_dir / args.train_file}")

    _write_sft_jsonl(Path(args.sft_train_file), train_examples)
    _write_sft_jsonl(Path(args.sft_val_file), val_examples)
    print(f"wrote {len(train_examples)} train / {len(val_examples)} val examples to {args.sft_train_file} / {args.sft_val_file}")

    if args.export_sft_only:
        return

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

    report = _evaluate_adapter(model, tokenizer, val_examples, max_new_tokens=args.max_new_tokens)
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


def _check_gpu_environment() -> None:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; CUDA QLoRA requires a CUDA GPU.")
    try:
        import bitsandbytes  # noqa: F401, PLC0415
    except ImportError as exc:
        raise RuntimeError("bitsandbytes is not installed; required for 4-bit QLoRA quantization.") from exc


class _Example:
    __slots__ = ("claim", "evidence_text", "gold_label", "prompt", "completion")

    def __init__(self, system_content: str, user_content: str, assistant_content: str) -> None:
        self.prompt = f"{system_content}\n\n{user_content}"
        self.completion = assistant_content

        claim_match = CLAIM_RE.search(user_content)
        self.claim = claim_match.group(1).strip() if claim_match else ""

        evidence_match = EVIDENCE_RE.search(user_content)
        self.evidence_text = evidence_match.group(1).strip() if evidence_match else ""

        verdict_match = VERDICT_RE.search(assistant_content)
        self.gold_label = verdict_match.group(1).upper() if verdict_match else UNPARSEABLE


def _load_examples(path: Path, *, limit: int) -> list[_Example]:
    rows = read_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    examples = []
    for row in rows:
        messages = {message["role"]: message["content"] for message in row["messages"]}
        examples.append(_Example(messages["system"], messages["user"], messages["assistant"]))
    return examples


def _write_sft_jsonl(path: Path, examples: list[_Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({"prompt": example.prompt, "completion": example.completion}) + "\n")


def _build_dataset(examples: list[_Example], tokenizer, max_seq_length: int):  # noqa: ANN001
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


def _evaluate_adapter(model, tokenizer, examples: list[_Example], *, max_new_tokens: int) -> dict:  # noqa: ANN001
    import torch  # noqa: PLC0415

    gold_labels: list[str] = []
    pred_labels: list[str] = []
    citation_valid: list[bool] = []
    unsupported_rates: list[float] = []
    latencies: list[float] = []
    error_examples: list[dict] = []

    model.eval()
    for example in examples:
        inputs = tokenizer(example.prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        started = perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        latencies.append(perf_counter() - started)
        response = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        verdict_match = VERDICT_RE.search(response)
        pred_label = verdict_match.group(1).upper() if verdict_match else UNPARSEABLE

        explanation_match = EXPLANATION_RE.search(response)
        explanation = explanation_match.group(1).strip() if explanation_match else response.strip()
        context = build_context(example.claim, [EvidenceSpan(doc_id="1", text=example.evidence_text)] if example.evidence_text else [])
        citation_result = check_citations(explanation, context)

        gold_labels.append(example.gold_label)
        pred_labels.append(pred_label)
        citation_valid.append(citation_result.valid)
        unsupported_rates.append(citation_result.unsupported_sentence_rate)

        if pred_label != example.gold_label and len(error_examples) < 5:
            error_examples.append(
                {
                    "claim": example.claim,
                    "gold_label": example.gold_label,
                    "predicted_label": pred_label,
                    "response": response.strip()[:300],
                }
            )

    from sklearn.metrics import f1_score  # noqa: PLC0415

    sample_size = len(examples)
    parseable_count = sum(1 for pred in pred_labels if pred != UNPARSEABLE)
    verdict_accuracy = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred) / sample_size

    macro_f1 = f1_score(gold_labels, pred_labels, labels=LABEL_ORDER, average="macro", zero_division=0)
    per_class_f1_scores = f1_score(gold_labels, pred_labels, labels=LABEL_ORDER, average=None, zero_division=0)
    per_class_f1 = dict(zip(LABEL_ORDER, (round(score, 4) for score in per_class_f1_scores)))

    return {
        "sample_size": sample_size,
        "verdict_accuracy": round(verdict_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class_f1,
        "parseable_verdicts": parseable_count,
        "parseable_rate": round(parseable_count / sample_size, 4),
        "citation_valid_rate": round(sum(citation_valid) / sample_size, 4),
        "unsupported_sentence_rate": round(sum(unsupported_rates) / sample_size, 4),
        "mean_latency_seconds": round(sum(latencies) / sample_size, 4),
        "top_error_examples": error_examples,
    }


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
