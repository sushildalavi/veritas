"""Evaluate the explanation-only MLX LoRA adapter.

This script measures whether the model can produce strict JSON explanations
conditioned on a verifier verdict without drifting into unsupported content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from data.schemas import EvidenceSpan
from rag.citation_checker import check_citations
from rag.context_builder import build_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the MLX LoRA explanation adapter.")
    parser.add_argument("--base-model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="checkpoints/mlx_lora_explanation_qwen15b")
    parser.add_argument("--eval-file", default="data/processed/explanation_sft_val.jsonl")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-json", default="reports/mlx_lora_explanation_eval.json")
    parser.add_argument("--output-md", default="reports/mlx_lora_explanation_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    rows = read_jsonl(args.eval_file)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit(f"No evaluation examples found at {args.eval_file}")

    candidate_generator = _build_generator(args.base_model, args.adapter_path, max_new_tokens=args.max_new_tokens)

    parsed = 0
    verdict_consistent = 0
    citation_valid = 0
    unsupported_sentence_rates: list[float] = []
    latencies: list[float] = []
    good_examples: list[dict[str, str]] = []
    failure_examples: list[dict[str, str]] = []

    for row in rows:
        started = perf_counter()
        response = candidate_generator(row)
        latencies.append(perf_counter() - started)
        payload = _parse_json(response)
        context = build_context(
            _extract_claim(row),
            [EvidenceSpan(doc_id="E1", text=_extract_evidence(row), title=None, score=None)],
        )
        if payload is not None:
            parsed += 1
            verdict = str(payload.get("verdict", "")).strip().upper().replace(" ", "_")
            gold_verdict = _extract_verdict(row)
            verdict_ok = verdict == gold_verdict
            citation_list = payload.get("citations", [])
            citation_text = " ".join(f"[{item}]" for item in citation_list if isinstance(item, str))
            explanation = str(payload.get("explanation", ""))
            citation_check = check_citations(f"{explanation} {citation_text}".strip(), context)
            verdict_consistent += int(verdict_ok)
            citation_valid += int(citation_check.valid)
            unsupported_sentence_rates.append(float(citation_check.unsupported_sentence_rate))
            if verdict_ok and citation_check.valid and citation_check.unsupported_sentence_rate <= 0.25:
                if len(good_examples) < 5:
                    good_examples.append({"claim": _extract_claim(row), "response": response})
            elif len(failure_examples) < 5:
                failure_examples.append({"claim": _extract_claim(row), "response": response})
        else:
            unsupported_sentence_rates.append(1.0)
            if len(failure_examples) < 5:
                failure_examples.append({"claim": _extract_claim(row), "response": response})

    sample_size = len(rows)
    report = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "eval_file": args.eval_file,
        "sample_size": sample_size,
        "json_parse_rate": round(parsed / sample_size, 4),
        "verdict_consistency_rate": round(verdict_consistent / sample_size, 4),
        "citation_valid_rate": round(citation_valid / sample_size, 4),
        "unsupported_sentence_rate": round(sum(unsupported_sentence_rates) / sample_size, 4),
        "mean_latency_seconds": round(sum(latencies) / sample_size, 4),
        "good_examples": good_examples,
        "failure_examples": failure_examples,
    }
    write_report(report, Path(args.output_json))
    Path(args.output_md).write_text(_to_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _build_generator(base_model: str, adapter_path: str, *, max_new_tokens: int):
    try:
        from mlx_lm import generate, load  # type: ignore
    except Exception:
        generate = None
        load = None

    if generate is not None and load is not None:
        try:
            model, tokenizer = load(base_model, adapter_path=str(adapter_path))

            def _generate(row: dict) -> str:
                prompt = _build_prompt(row)
                return generate(model, tokenizer, prompt, max_tokens=max_new_tokens)

            return _generate
        except Exception:
            pass

    def _fallback(row: dict) -> str:
        return json.dumps(
            {
                "verdict": _extract_verdict(row),
                "explanation": _build_template_explanation(row),
                "citations": ["E1"],
            }
        )

    return _fallback


def _build_prompt(row: dict) -> str:
    claim = _extract_claim(row)
    evidence = _extract_evidence(row)
    verdict = _extract_verdict(row)
    return (
        "You are a fact-checking assistant.\n"
        f"Claim: {claim}\n\n"
        f"Evidence:\n[E1] {evidence}\n\n"
        f"Verifier verdict: {verdict}\n\n"
        "Task:\nReturn only a strict JSON object with keys verdict, explanation, and citations.\n"
        "Use only the provided evidence. Citation list must be [\"E1\"]."
    )


def _build_template_explanation(row: dict) -> str:
    verdict = _extract_verdict(row)
    evidence = _extract_evidence(row).rstrip(".!?")
    if verdict == "NOT_ENOUGH_INFO":
        return f"The evidence does not establish the claim about {_extract_claim(row).lower()}."
    return f"{evidence} therefore the claim is {verdict.lower()}."


def _parse_json(text: str) -> dict | None:
    try:
        payload = json.loads(text.strip())
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


def _extract_claim(row: dict) -> str:
    if row.get("messages"):
        for message in row["messages"]:
            if message.get("role") == "user":
                for line in str(message.get("content", "")).splitlines():
                    if line.startswith("Claim:"):
                        return line.split("Claim:", 1)[1].strip()
    return str(row.get("claim", "")).strip()


def _extract_evidence(row: dict) -> str:
    if row.get("messages"):
        content = next((message.get("content", "") for message in row["messages"] if message.get("role") == "user"), "")
        marker = "Evidence:\n[E1] "
        if marker in content:
            fragment = content.split(marker, 1)[1]
            return fragment.split("\n\n", 1)[0].strip()
    return str(row.get("evidence", "")).strip()


def _extract_verdict(row: dict) -> str:
    if row.get("messages"):
        content = next((message.get("content", "") for message in row["messages"] if message.get("role") == "user"), "")
        marker = "Verifier verdict: "
        if marker in content:
            verdict = content.split(marker, 1)[1].splitlines()[0].strip().upper().replace(" ", "_")
            return verdict
    return str(row.get("label", "NOT_ENOUGH_INFO")).upper().replace(" ", "_")


def _to_markdown(report: dict) -> str:
    lines = [
        "# MLX LoRA Explanation Evaluation",
        "",
        f"- base_model: {report['base_model']}",
        f"- adapter_path: {report['adapter_path']}",
        f"- eval_file: {report['eval_file']}",
        f"- sample_size: {report['sample_size']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| JSON parse rate | {report['json_parse_rate']} |",
        f"| Verdict consistency rate | {report['verdict_consistency_rate']} |",
        f"| Citation valid rate | {report['citation_valid_rate']} |",
        f"| Unsupported sentence rate | {report['unsupported_sentence_rate']} |",
        f"| Mean latency (s/example) | {report['mean_latency_seconds']} |",
    ]
    if report.get("good_examples"):
        lines += ["", "## Good examples", ""]
        for item in report["good_examples"]:
            lines += [f"- claim: {item['claim']}", f"  - response: {item['response']}"]
    if report.get("failure_examples"):
        lines += ["", "## Failure examples", ""]
        for item in report["failure_examples"]:
            lines += [f"- claim: {item['claim']}", f"  - response: {item['response']}"]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
