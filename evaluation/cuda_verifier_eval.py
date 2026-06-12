"""Shared dataset loading and evaluation for CUDA verifier adapters.

Used by ``scripts/train_cuda_qlora.py`` and ``scripts/train_cuda_dpo.py`` so
both load ``data/processed/mlx_lora/{train,valid}.jsonl`` the same way and
report the same metrics (verdict accuracy, macro F1, per-class F1, citation
valid rate, unsupported-sentence rate, mean latency) as
``evaluation/mlx_lora_eval.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter

from data.schemas import EvidenceSpan
from evaluation.mlx_lora_eval import EXPLANATION_RE, LABEL_ORDER, UNPARSEABLE, VERDICT_RE
from evaluation.sample_benchmarks import read_jsonl
from rag import build_context, check_citations

CLAIM_RE = re.compile(r"Claim: (.*?)\nEvidence:", re.DOTALL)
EVIDENCE_RE = re.compile(r"Evidence: \[1\] (.*?)\nClassify", re.DOTALL)


class Example:
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


def load_examples(path: Path, *, limit: int = 0) -> list[Example]:
    rows = read_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    examples = []
    for row in rows:
        messages = {message["role"]: message["content"] for message in row["messages"]}
        examples.append(Example(messages["system"], messages["user"], messages["assistant"]))
    return examples


def write_sft_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({"prompt": example.prompt, "completion": example.completion}) + "\n")


def check_cuda_environment(extra_packages: tuple[str, ...] = ()) -> None:
    """Raise ``RuntimeError`` if CUDA or any of ``extra_packages`` are unavailable."""
    import importlib

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; this step requires a CUDA GPU.")
    for package in extra_packages:
        try:
            importlib.import_module(package)
        except ImportError as exc:
            raise RuntimeError(f"{package} is not installed; required for this step.") from exc


def evaluate_adapter(model, tokenizer, examples: list[Example], *, max_new_tokens: int = 160) -> dict:  # noqa: ANN001
    """Evaluate a CUDA causal-LM adapter on ``examples`` (from ``load_examples``).

    Mirrors ``evaluation.mlx_lora_eval.evaluate_adapter`` so CUDA QLoRA/DPO
    reports use the same metric definitions as the MLX LoRA reports, plus
    ``verdict_consistency_rate`` (citation precision >= 0.5 and unsupported
    sentence rate <= 0.5) for before/after DPO comparisons.
    """
    import torch  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415

    gold_labels: list[str] = []
    pred_labels: list[str] = []
    citation_valid: list[bool] = []
    unsupported_rates: list[float] = []
    verdict_consistent: list[bool] = []
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
        verdict_consistent.append(citation_result.verdict_consistency)

        if pred_label != example.gold_label and len(error_examples) < 5:
            error_examples.append(
                {
                    "claim": example.claim,
                    "gold_label": example.gold_label,
                    "predicted_label": pred_label,
                    "response": response.strip()[:300],
                }
            )

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
        "verdict_consistency_rate": round(sum(verdict_consistent) / sample_size, 4),
        "mean_latency_seconds": round(sum(latencies) / sample_size, 4),
        "top_error_examples": error_examples,
    }
