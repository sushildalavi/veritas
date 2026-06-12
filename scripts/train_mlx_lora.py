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
import re
import types
from pathlib import Path
from time import perf_counter

import yaml
from sklearn.metrics import f1_score

from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from rag import build_context, check_citations

DISPLAY_LABEL = {"SUPPORTED": "SUPPORTED", "REFUTED": "REFUTED", "NOT_ENOUGH_INFO": "NOT ENOUGH INFO"}
SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Given a claim and evidence, classify "
    "the claim as SUPPORTED, REFUTED, or NOT ENOUGH INFO and explain your "
    "verdict using only the provided evidence, citing evidence ids in "
    "brackets."
)

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
VERDICT_RE = re.compile(r"Verdict:\s*(SUPPORTED|REFUTED|NOT ENOUGH INFO)", re.IGNORECASE)
EXPLANATION_RE = re.compile(r"Explanation:\s*(.*?)(?:\nCitation:|$)", re.DOTALL | re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune an MLX Qwen2.5 model with LoRA for fact verification.")
    parser.add_argument("--config", default="configs/mlx_lora_qwen05b.yaml")
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

    from mlx_lm import generate, load  # noqa: PLC0415
    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser as build_lora_parser, run as run_lora  # noqa: PLC0415

    config_path = Path(args.config)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if args.iters is not None:
        config["iters"] = args.iters

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
    model, tokenizer = load(base_model, adapter_path=str(adapter_path))

    rows = read_jsonl(Path(args.data_dir) / args.eval_file)
    if args.max_eval_examples > 0:
        rows = rows[: args.max_eval_examples]
    if not rows:
        raise SystemExit(f"No evaluation examples found at {Path(args.data_dir) / args.eval_file}")

    gold_labels: list[str] = []
    pred_labels: list[str | None] = []
    citation_valid: list[bool] = []
    unsupported_rates: list[float] = []
    latencies: list[float] = []

    for row in rows:
        claim = str(row.get("claim", ""))
        evidence_text = str(row.get("evidence", ""))
        gold_label = DISPLAY_LABEL[str(row.get("label", "NOT_ENOUGH_INFO"))]

        context = build_context(claim, [EvidenceSpan(doc_id="1", text=evidence_text)] if evidence_text else [])
        evidence_block = context.format_block() or "(no evidence provided)"
        user_content = (
            f"Claim: {claim}\n"
            f"Evidence: {evidence_block}\n"
            "Classify the claim and explain using only the evidence."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        started = perf_counter()
        response = generate(model, tokenizer, prompt, max_tokens=args.max_new_tokens)
        latencies.append(perf_counter() - started)

        verdict_match = VERDICT_RE.search(response)
        pred_label = verdict_match.group(1).upper() if verdict_match else None

        explanation_match = EXPLANATION_RE.search(response)
        explanation = explanation_match.group(1).strip() if explanation_match else response.strip()
        citation_result = check_citations(explanation, context)

        gold_labels.append(gold_label)
        pred_labels.append(pred_label)
        citation_valid.append(citation_result.valid)
        unsupported_rates.append(citation_result.unsupported_sentence_rate)

    sample_size = len(rows)
    parseable = [pred for pred in pred_labels if pred is not None]
    verdict_accuracy = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred) / sample_size

    macro_f1 = None
    if len(parseable) == sample_size:
        macro_f1 = f1_score(gold_labels, pred_labels, labels=LABEL_ORDER, average="macro")

    report = {
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "fine_tune_type": lora_args.get("fine_tune_type"),
        "lora_iters": lora_args.get("iters"),
        "train_data": lora_args.get("data"),
        "eval_file": str(Path(args.data_dir) / args.eval_file),
        "sample_size": sample_size,
        "verdict_accuracy": round(verdict_accuracy, 4),
        "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
        "macro_f1_note": None if macro_f1 is not None else "not all verdicts were parseable",
        "citation_valid_rate": round(sum(citation_valid) / sample_size, 4),
        "unsupported_sentence_rate": round(sum(unsupported_rates) / sample_size, 4),
        "mean_latency_seconds": round(sum(latencies) / sample_size, 4),
        "parseable_verdicts": len(parseable),
    }

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(f"mlx lora eval: {report}")


def _to_markdown(report: dict) -> str:
    lines = [
        "# MLX LoRA Verifier Evaluation",
        "",
        f"- base_model: {report['base_model']}",
        f"- adapter_path: {report['adapter_path']}",
        f"- fine_tune_type: {report['fine_tune_type']}",
        f"- lora_iters: {report['lora_iters']}",
        f"- train_data: {report['train_data']}",
        f"- eval_file: {report['eval_file']}",
        f"- sample_size: {report['sample_size']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Verdict accuracy | {report['verdict_accuracy']} |",
        f"| Macro F1 | {report['macro_f1'] if report['macro_f1'] is not None else 'n/a (' + str(report['macro_f1_note']) + ')'} |",
        f"| Citation valid rate | {report['citation_valid_rate']} |",
        f"| Unsupported sentence rate | {report['unsupported_sentence_rate']} |",
        f"| Mean latency (s/example) | {report['mean_latency_seconds']} |",
        f"| Parseable verdicts | {report['parseable_verdicts']} / {report['sample_size']} |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
