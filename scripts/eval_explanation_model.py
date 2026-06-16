"""Evaluate explanation models on grounded explanation artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from rag import build_context, check_citations

DECISION_RE = re.compile(r"Decision:\s*(SUPPORTED|REFUTED|NOT_ENOUGH_INFO)", re.IGNORECASE)
EXPLANATION_RE = re.compile(r"Explanation:\s*(.*?)(?:\nCitations:|$)", re.DOTALL | re.IGNORECASE)
CITATIONS_RE = re.compile(r"Citations:\s*(\[.*\])", re.DOTALL | re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate explanation models.")
    parser.add_argument("--input-file", default="data/explanations/sft_test.jsonl")
    parser.add_argument("--base-model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="adapters/mlx_qwen_veritas_lora")
    parser.add_argument("--transformers-model", default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--report-json", default="reports/explanation_model_eval.json")
    parser.add_argument("--report-md", default="reports/explanation_model_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    rows = read_jsonl(Path(args.input_file))
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit(f"No examples found in {args.input_file}")

    report = {
        "input_file": args.input_file,
        "sample_size": len(rows),
        "backends": [],
    }

    report["backends"].append(_evaluate_mlx_backend("mlx_base", args.base_model, None, rows))
    adapter_path = Path(args.adapter_path)
    if adapter_path.exists():
        report["backends"].append(_evaluate_mlx_backend("mlx_adapter", args.base_model, adapter_path, rows))
    else:
        report["backends"].append(
            {
                "backend": "mlx_adapter",
                "status": "skipped",
                "reason": f"adapter path not found: {adapter_path}",
                "model": args.base_model,
                "adapter_path": str(adapter_path),
            }
        )

    if args.transformers_model:
        report["backends"].append(_evaluate_transformers_backend(args.transformers_model, rows))

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote explanation model evaluation to {args.report_json}")


def _evaluate_mlx_backend(backend: str, base_model: str, adapter_path: Path | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from mlx_lm import generate, load  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "backend": backend,
            "status": "skipped",
            "reason": f"mlx-lm unavailable: {exc}",
            "model": base_model,
            "adapter_path": str(adapter_path) if adapter_path else None,
        }

    try:
        if adapter_path is None:
            model, tokenizer = load(base_model)
        else:
            model, tokenizer = load(base_model, adapter_path=str(adapter_path))
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "backend": backend,
            "status": "skipped",
            "reason": f"model load failed: {exc}",
            "model": base_model,
            "adapter_path": str(adapter_path) if adapter_path else None,
        }

    def _mlx_generate(prompt: str) -> str:
        raw = str(generate(model, tokenizer, prompt, max_tokens=200))
        # strip prompt leakage after end token
        for stop in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
            if stop in raw:
                raw = raw[:raw.index(stop)]
        return raw.strip()

    return _evaluate_model(backend, base_model, adapter_path, rows, _mlx_generate)


def _evaluate_transformers_backend(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "backend": "transformers",
            "status": "skipped",
            "reason": f"transformers unavailable: {exc}",
            "model": model_name,
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "backend": "transformers",
            "status": "skipped",
            "reason": f"model load failed: {exc}",
            "model": model_name,
        }

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=160)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return _evaluate_model("transformers", model_name, None, rows, generate)


def _evaluate_model(
    backend: str,
    model_name: str,
    adapter_path: Path | None,
    rows: list[dict[str, Any]],
    generate_text: Callable[[str], str],
) -> dict[str, Any]:
    exact_samples: list[dict[str, Any]] = []
    format_correct = 0
    decision_consistent = 0
    citation_presence = 0
    citation_precision_sum = 0.0
    unsupported_rate_sum = 0.0
    explanation_length_sum = 0.0
    total_runtime = 0.0

    for row in rows:
        prompt = str(row.get("prompt", ""))
        verifier_label = str(row.get("verifier_label", "NOT_ENOUGH_INFO"))
        evidence_passages = _to_evidence_spans(row)
        output, elapsed = _safe_generate(generate_text, prompt)
        total_runtime += elapsed

        parsed = _parse_output(output)
        explanation = parsed.get("explanation", "").strip()
        citations = parsed.get("citations", [])
        decision = parsed.get("decision", "")
        format_ok = bool(decision) and bool(explanation) and isinstance(citations, list)
        format_correct += int(format_ok)
        decision_consistent += int(str(decision) == verifier_label)
        citation_presence += int(bool(citations))
        citation_precision_sum += _citation_precision(citations, evidence_passages)
        unsupported_rate_sum += _unsupported_rate(explanation, row)
        explanation_length_sum += len(explanation.split())

        if len(exact_samples) < 10:
            exact_samples.append(
                {
                    "claim_id": row.get("claim_id"),
                    "verifier_label": verifier_label,
                    "prompt": prompt,
                    "output": output,
                    "parsed": parsed,
                    "format_correct": format_ok,
                    "citation_precision": _citation_precision(citations, evidence_passages),
                    "unsupported_sentence_rate": _unsupported_rate(explanation, row),
                }
            )

    sample_size = len(rows)
    return {
        "backend": backend,
        "status": "trained" if adapter_path else "measured",
        "model": model_name,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "sample_size": sample_size,
        "format_correctness": round(format_correct / sample_size, 4),
        "decision_label_consistency": round(decision_consistent / sample_size, 4),
        "citation_presence": round(citation_presence / sample_size, 4),
        "citation_precision": round(citation_precision_sum / sample_size, 4),
        "unsupported_claim_rate": round(unsupported_rate_sum / sample_size, 4),
        "average_explanation_length": round(explanation_length_sum / sample_size, 2),
        "mean_generation_time_seconds": round(total_runtime / sample_size, 4),
        "exact_sample_outputs": exact_samples,
    }


def _safe_generate(generate_text: Callable[[str], str], prompt: str) -> tuple[str, float]:
    try:
        started = perf_counter()
        value = str(generate_text(prompt))
        return value, perf_counter() - started
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"Decision: NOT_ENOUGH_INFO\nExplanation: generation failed: {exc}\nCitations: []", 0.0


def _parse_output(output: str) -> dict[str, Any]:
    decision_match = DECISION_RE.search(output)
    explanation_match = EXPLANATION_RE.search(output)
    citations_match = CITATIONS_RE.search(output)
    citations: list[str] = []
    if citations_match:
        try:
            parsed = json.loads(citations_match.group(1))
            if isinstance(parsed, list):
                citations = [str(item) for item in parsed]
        except json.JSONDecodeError:
            citations = []
    return {
        "decision": decision_match.group(1).upper() if decision_match else "",
        "explanation": explanation_match.group(1).strip() if explanation_match else "",
        "citations": citations,
    }


def _to_evidence_spans(row: dict[str, Any]) -> list[EvidenceSpan]:
    spans = []
    for item in row.get("evidence_passages", []):
        if not isinstance(item, dict):
            continue
        spans.append(
            EvidenceSpan(
                doc_id=str(item.get("doc_id", "")),
                text=str(item.get("text", "")),
                title=item.get("title"),
                score=item.get("score"),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return spans


def _citation_precision(citations: list[str], evidence_passages: list[EvidenceSpan]) -> float:
    if not citations:
        return 0.0
    evidence_ids = {span.doc_id for span in evidence_passages}
    matches = sum(1 for citation in citations if citation in evidence_ids)
    return round(matches / len(citations), 4)


def _unsupported_rate(explanation: str, row: dict[str, Any]) -> float:
    evidence_spans = _to_evidence_spans(row)
    if not evidence_spans:
        return 1.0 if explanation.strip() else 0.0
    context = build_context(str(row.get("claim", "")), evidence_spans)
    return float(check_citations(explanation, context).unsupported_sentence_rate)


def _to_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Explanation Model Evaluation",
        "",
        f"- input_file: {report['input_file']}",
        f"- sample_size: {report['sample_size']}",
        "",
        "| Backend | Status | Model | Adapter | Format | Decision | Citations | Precision | Unsupported | Length |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backend in report["backends"]:
        if backend.get("status") == "skipped":
            lines.append(
                f"| {backend['backend']} | skipped | {backend.get('model', '')} | {backend.get('adapter_path', '')} | - | - | - | - | - | - |"
            )
            lines.append(f"\nReason: {backend['reason']}\n")
        else:
            lines.append(
                f"| {backend['backend']} | {backend['status']} | {backend['model']} | {backend.get('adapter_path') or '-'} | "
                f"{backend['format_correctness']} | {backend['decision_label_consistency']} | {backend['citation_presence']} | "
                f"{backend['citation_precision']} | {backend['unsupported_claim_rate']} | {backend['average_explanation_length']} |"
            )
            lines.append("")
            lines.append(f"## {backend['backend']} Sample Outputs")
            lines.append("")
            for sample in backend["exact_sample_outputs"]:
                lines += [
                    f"- claim_id: {sample['claim_id']}",
                    f"  - verifier_label: {sample['verifier_label']}",
                    f"  - format_correct: {sample['format_correct']}",
                    f"  - citation_precision: {sample['citation_precision']}",
                    f"  - unsupported_sentence_rate: {sample['unsupported_sentence_rate']}",
                    "  - output:",
                    "```text",
                    _to_text(sample["output"]),
                    "```",
                ]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
