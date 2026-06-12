"""Final citation-faithfulness comparison across explanation generators.

Measures citation faithfulness of the deployed template explanation
generator (``rag.generate_template_explanation``) driven by the clean,
class-weighted DistilRoBERTa verifier
(``checkpoints/transformer_verifier_clean``) on a real validation sample.

QLoRA and DPO explanation generators are not included with measured numbers
because their training is blocked on this machine (no CUDA GPU /
bitsandbytes - see ``reports/qlora_BLOCKED_GPU_REQUIRED.md`` and
``reports/dpo_BLOCKED_QLORA_REQUIRED.md``). Their rows are recorded as
``status: "not trained - GPU required"`` rather than fabricated.

Outputs:
  reports/faithfulness_comparison.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from models.deberta_verifier import VerificationResult
from rag import build_context, check_citations, generate_template_explanation

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
DISPLAY_LABEL = {"SUPPORTED": "SUPPORTED", "REFUTED": "REFUTED", "NOT_ENOUGH_INFO": "NOT ENOUGH INFO"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare explanation-generator citation faithfulness.")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_verifier_clean")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--val-file", default="verifier_val.jsonl")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report-json", default="reports/faithfulness_comparison.json")
    parser.add_argument("--report-md", default="reports/faithfulness_comparison.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)

    rows = read_jsonl(data_dir / args.val_file)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit(f"No validation examples found at {data_dir / args.val_file}")

    tokenizer, model = _load_transformer(args.checkpoint)
    model.eval()

    claims = [str(row.get("claim", "")) for row in rows]
    evidence_texts = [str(row.get("evidence", "")) for row in rows]
    gold_labels = [str(row.get("label", "NOT_ENOUGH_INFO")) for row in rows]

    texts = [f"Claim: {claim}\nEvidence: {evidence}" for claim, evidence in zip(claims, evidence_texts)]

    started = perf_counter()
    preds = _predict(model, tokenizer, texts, args.max_length, args.batch_size)
    runtime_seconds = round(perf_counter() - started, 3)

    template_summary = _evaluate_template(claims, evidence_texts, gold_labels, preds)
    template_summary["example_count"] = len(rows)
    template_summary["runtime_seconds"] = runtime_seconds

    report = {
        "checkpoint": args.checkpoint,
        "data_file": str(data_dir / args.val_file),
        "generators": {
            "template": {
                "status": "measured",
                "description": "rag.generate_template_explanation driven by checkpoints/transformer_verifier_clean predictions",
                **template_summary,
            },
            "qlora": {
                "status": "not trained - GPU required",
                "description": "TinyLlama + QLoRA explanation generator (see reports/qlora_BLOCKED_GPU_REQUIRED.md)",
            },
            "dpo": {
                "status": "not trained - QLoRA required",
                "description": "DPO-aligned explanation generator (see reports/dpo_BLOCKED_QLORA_REQUIRED.md)",
            },
        },
    }

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(f"template faithfulness: {template_summary}")


def _load_transformer(checkpoint: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    return tokenizer, model


def _predict(model, tokenizer, texts: list[str], max_length: int, batch_size: int) -> list[int]:  # noqa: ANN001
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            logits = model(**encoded).logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
    return predictions


def _evaluate_template(
    claims: list[str], evidence_texts: list[str], gold_labels: list[str], preds: list[int]
) -> dict[str, float]:
    rows: list[dict[str, object]] = []
    correct = 0
    for claim, evidence_text, gold_label, pred in zip(claims, evidence_texts, gold_labels, preds):
        predicted_label = LABEL_ORDER[pred]
        if predicted_label == gold_label:
            correct += 1
        context = build_context(claim, [EvidenceSpan(doc_id="1", text=evidence_text)] if evidence_text else [])
        verification = VerificationResult(verdict=DISPLAY_LABEL[predicted_label], confidence=1.0)
        explanation_output = generate_template_explanation(context, verification)
        citation_result = check_citations(explanation_output.explanation, context)
        rows.append(
            {
                "citation_valid": citation_result.valid,
                "citation_precision": citation_result.citation_precision,
                "unsupported_sentence_rate": citation_result.unsupported_sentence_rate,
                "verdict_consistency": citation_result.verdict_consistency,
            }
        )

    return {
        "verifier_accuracy": correct / len(rows) if rows else 0.0,
        "citation_valid_rate": mean(1.0 if row["citation_valid"] else 0.0 for row in rows) if rows else 0.0,
        "mean_citation_precision": mean(float(row["citation_precision"]) for row in rows) if rows else 0.0,
        "mean_unsupported_sentence_rate": (
            mean(float(row["unsupported_sentence_rate"]) for row in rows) if rows else 0.0
        ),
        "verdict_consistency_rate": mean(1.0 if row["verdict_consistency"] else 0.0 for row in rows) if rows else 0.0,
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Final Faithfulness Comparison",
        "",
        f"- Verifier checkpoint: `{report['checkpoint']}`",
        f"- Data file: `{report['data_file']}`",
        "",
        "| generator | status | example_count | verifier_accuracy | citation_valid_rate | mean_citation_precision | mean_unsupported_sentence_rate | verdict_consistency_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, data in report["generators"].items():
        if data["status"] == "measured":
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        data["status"],
                        str(data["example_count"]),
                        f"{data['verifier_accuracy']:.3f}",
                        f"{data['citation_valid_rate']:.3f}",
                        f"{data['mean_citation_precision']:.3f}",
                        f"{data['mean_unsupported_sentence_rate']:.3f}",
                        f"{data['verdict_consistency_rate']:.3f}",
                    ]
                )
                + " |"
            )
        else:
            lines.append(f"| {name} | {data['status']} | - | - | - | - | - | - |")
    lines.extend(["", "## Notes", ""])
    for name, data in report["generators"].items():
        lines.append(f"- **{name}**: {data['description']} ({data['status']})")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
