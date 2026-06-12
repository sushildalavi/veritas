"""Shared evaluation logic for the Apple Silicon MLX LoRA verifier adapter.

Used by ``scripts/eval_mlx_lora.py`` (standalone evaluation) and
``scripts/train_mlx_lora.py`` (post-training evaluation) so both report the
same metrics in the same format.
"""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from sklearn.metrics import f1_score

from data.schemas import EvidenceSpan
from rag import build_context, check_citations

DISPLAY_LABEL = {"SUPPORTED": "SUPPORTED", "REFUTED": "REFUTED", "NOT_ENOUGH_INFO": "NOT ENOUGH INFO"}
SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Given a claim and evidence, classify "
    "the claim as SUPPORTED, REFUTED, or NOT ENOUGH INFO and explain your "
    "verdict using only the provided evidence, citing evidence ids in "
    "brackets."
)

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
UNPARSEABLE = "UNPARSEABLE"
VERDICT_RE = re.compile(r"Verdict:\s*(SUPPORTED|REFUTED|NOT ENOUGH INFO)", re.IGNORECASE)
EXPLANATION_RE = re.compile(r"Explanation:\s*(.*?)(?:\nCitation:|$)", re.DOTALL | re.IGNORECASE)


def build_prompt(tokenizer: Any, claim: str, evidence_text: str) -> tuple[Any, Any]:
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
    return prompt, context


def evaluate_adapter(
    base_model: str,
    adapter_path: str,
    rows: list[dict],
    *,
    max_new_tokens: int = 160,
    max_error_examples: int = 5,
    model: Any = None,
    tokenizer: Any = None,
) -> dict:
    """Evaluate an MLX LoRA adapter on ``rows`` (verifier_*.jsonl-shaped dicts).

    If ``model``/``tokenizer`` are not provided, loads
    ``base_model``+``adapter_path`` via ``mlx_lm.load``.
    """
    from mlx_lm import generate, load  # noqa: PLC0415

    if model is None or tokenizer is None:
        model, tokenizer = load(base_model, adapter_path=str(adapter_path))

    gold_labels: list[str] = []
    pred_labels: list[str] = []
    citation_valid: list[bool] = []
    unsupported_rates: list[float] = []
    latencies: list[float] = []
    error_examples: list[dict] = []

    for row in rows:
        claim = str(row.get("claim", ""))
        evidence_text = str(row.get("evidence", ""))
        gold_label = DISPLAY_LABEL[str(row.get("label", "NOT_ENOUGH_INFO"))]

        prompt, context = build_prompt(tokenizer, claim, evidence_text)

        started = perf_counter()
        response = generate(model, tokenizer, prompt, max_tokens=max_new_tokens)
        latencies.append(perf_counter() - started)

        verdict_match = VERDICT_RE.search(response)
        pred_label = verdict_match.group(1).upper() if verdict_match else UNPARSEABLE

        explanation_match = EXPLANATION_RE.search(response)
        explanation = explanation_match.group(1).strip() if explanation_match else response.strip()
        citation_result = check_citations(explanation, context)

        gold_labels.append(gold_label)
        pred_labels.append(pred_label)
        citation_valid.append(citation_result.valid)
        unsupported_rates.append(citation_result.unsupported_sentence_rate)

        if pred_label != gold_label and len(error_examples) < max_error_examples:
            error_examples.append(
                {
                    "claim": claim,
                    "gold_label": gold_label,
                    "predicted_label": pred_label,
                    "response": response.strip()[:300],
                }
            )

    sample_size = len(rows)
    parseable_count = sum(1 for pred in pred_labels if pred != UNPARSEABLE)
    verdict_accuracy = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred) / sample_size

    macro_f1 = f1_score(gold_labels, pred_labels, labels=LABEL_ORDER, average="macro", zero_division=0)
    per_class_f1_scores = f1_score(gold_labels, pred_labels, labels=LABEL_ORDER, average=None, zero_division=0)
    per_class_f1 = dict(zip(LABEL_ORDER, (round(score, 4) for score in per_class_f1_scores)))

    return {
        "base_model": base_model,
        "adapter_path": str(adapter_path),
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


def to_markdown(report: dict, *, title: str = "MLX LoRA Verifier Evaluation") -> str:
    lines = [
        f"# {title}",
        "",
        f"- base_model: {report['base_model']}",
        f"- adapter_path: {report['adapter_path']}",
    ]
    for key in ("fine_tune_type", "lora_iters", "train_data", "eval_file"):
        if key in report:
            lines.append(f"- {key}: {report[key]}")
    lines += [
        f"- sample_size: {report['sample_size']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Verdict accuracy | {report['verdict_accuracy']} |",
        f"| Macro F1 | {report['macro_f1']} |",
        f"| Citation valid rate | {report['citation_valid_rate']} |",
        f"| Unsupported sentence rate | {report['unsupported_sentence_rate']} |",
        f"| Mean latency (s/example) | {report['mean_latency_seconds']} |",
        f"| Parseable verdicts | {report['parseable_verdicts']} / {report['sample_size']} ({report['parseable_rate']}) |",
        "",
        "## Per-class F1",
        "",
        "| Label | F1 |",
        "| --- | ---: |",
    ]
    for label in LABEL_ORDER:
        lines.append(f"| {label} | {report['per_class_f1'].get(label, 0.0)} |")

    if report.get("top_error_examples"):
        lines += ["", "## Top error examples", ""]
        for example in report["top_error_examples"]:
            lines += [
                f"- claim: {example['claim']}",
                f"  - gold: {example['gold_label']}, predicted: {example['predicted_label']}",
                f"  - response: {example['response']}",
            ]
    lines.append("")
    return "\n".join(lines)
