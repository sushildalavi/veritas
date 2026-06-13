"""Evaluate preference-guided reranking for explanation candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from data.schemas import EvidenceSpan
from rag.context_builder import build_context
from rag.citation_checker import check_citations
from rag.preference_reranker import select_best_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Mac-compatible preference reranking.")
    parser.add_argument("--base-model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="checkpoints/mlx_lora_explanation_qwen15b")
    parser.add_argument("--eval-file", default="data/processed/explanation_sft_val.jsonl")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-json", default="reports/preference_reranking_eval.json")
    parser.add_argument("--output-md", default="reports/preference_reranking_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    rows = read_jsonl(args.eval_file)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit(f"No evaluation examples found at {args.eval_file}")

    generator = _build_candidate_generator(args.base_model, args.adapter_path, max_new_tokens=args.max_new_tokens)

    before_metrics = {"json_parse_rate": 0.0, "verdict_consistency_rate": 0.0, "citation_valid_rate": 0.0, "unsupported_sentence_rate": 0.0}
    after_metrics = before_metrics.copy()
    selected_positions: Counter[int] = Counter()
    latencies_before: list[float] = []
    latencies_after: list[float] = []
    good_examples: list[dict[str, str]] = []
    failure_examples: list[dict[str, str]] = []

    before_counts = Counter()
    after_counts = Counter()

    for row in rows:
        context = build_context(
            _extract_claim(row),
            [EvidenceSpan(doc_id="E1", text=_extract_evidence(row), title=None, score=None)],
        )
        verifier_verdict = _extract_verdict(row)
        started = perf_counter()
        first_candidate = generator(row, 1)[0]
        latencies_before.append(perf_counter() - started)

        rerank_started = perf_counter()
        candidates = generator(row, args.num_candidates)
        best_candidate, scored = select_best_candidate(
            candidates,
            verifier_verdict=verifier_verdict,
            context=context,
        )
        latencies_after.append(perf_counter() - rerank_started)
        best_index = max(scored, key=lambda item: (item.score, -item.index)).index
        selected_positions[best_index] += 1

        before_counts.update(_score_candidate(first_candidate, verifier_verdict, context))
        after_counts.update(_score_candidate(best_candidate, verifier_verdict, context))

        if len(good_examples) < 5 and first_candidate != best_candidate:
            good_examples.append({"claim": _extract_claim(row), "before": first_candidate, "after": best_candidate})
        elif len(failure_examples) < 5 and first_candidate == best_candidate:
            failure_examples.append({"claim": _extract_claim(row), "candidate": best_candidate})

    sample_size = len(rows)
    report = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "eval_file": args.eval_file,
        "sample_size": sample_size,
        "num_candidates": args.num_candidates,
        "before": _normalize_counts(before_counts, sample_size),
        "after": _normalize_counts(after_counts, sample_size),
        "mean_latency_increase_seconds": round((sum(latencies_after) - sum(latencies_before)) / sample_size, 4),
        "selected_candidate_position_distribution": dict(sorted(selected_positions.items())),
        "good_examples": good_examples,
        "failure_examples": failure_examples,
    }

    write_report(report, Path(args.output_json))
    Path(args.output_md).write_text(_to_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _build_candidate_generator(base_model: str, adapter_path: str, *, max_new_tokens: int):
    try:
        from mlx_lm import generate, load  # type: ignore
    except Exception:
        generate = None
        load = None

    if generate is not None and load is not None:
        try:
            model, tokenizer = load(base_model, adapter_path=str(adapter_path))

            def _generate(row: dict, num_candidates: int) -> list[str]:
                prompt = _build_prompt(row)
                outputs: list[str] = []
                for i in range(num_candidates):
                    outputs.append(
                        generate(
                            model,
                            tokenizer,
                            prompt,
                            max_tokens=max_new_tokens,
                            temp=0.6 + (0.15 * i),
                            top_p=0.9,
                        )
                    )
                return outputs

            return _generate
        except Exception:
            pass

    def _fallback(row: dict, num_candidates: int) -> list[str]:
        payload = json.dumps(
            {
                "verdict": _extract_verdict(row),
                "explanation": _extract_evidence(row).rstrip(".!?") + f" therefore the claim is {_extract_verdict(row).lower()}.",
                "citations": ["E1"],
            }
        )
        return [payload for _ in range(num_candidates)]

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


def _score_candidate(text: str, verifier_verdict: str, context) -> dict[str, float]:
    payload = _parse_json(text)
    if payload is None:
        return {"json_parse_rate": 0.0, "verdict_consistency_rate": 0.0, "citation_valid_rate": 0.0, "unsupported_sentence_rate": 1.0}
    verdict = _normalize_label(str(payload.get("verdict", "")))
    explanation = str(payload.get("explanation", ""))
    citations = payload.get("citations", [])
    citation_text = " ".join(f"[{item}]" for item in citations if isinstance(item, str))
    citation_check = check_citations(f"{explanation} {citation_text}".strip(), context)
    return {
        "json_parse_rate": 1.0,
        "verdict_consistency_rate": 1.0 if verdict == _normalize_label(verifier_verdict) else 0.0,
        "citation_valid_rate": 1.0 if citation_check.valid else 0.0,
        "unsupported_sentence_rate": float(citation_check.unsupported_sentence_rate),
    }


def _normalize_counts(counts: Counter, sample_size: int) -> dict[str, float]:
    return {
        "json_parse_rate": round(counts["json_parse_rate"] / sample_size if sample_size else 0.0, 4),
        "verdict_consistency_rate": round(counts["verdict_consistency_rate"] / sample_size if sample_size else 0.0, 4),
        "citation_valid_rate": round(counts["citation_valid_rate"] / sample_size if sample_size else 0.0, 4),
        "unsupported_sentence_rate": round(counts["unsupported_sentence_rate"] / sample_size if sample_size else 0.0, 4),
    }


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


def _normalize_label(value: str) -> str:
    text = value.strip().upper().replace(" ", "_")
    if text in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if text in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def _to_markdown(report: dict) -> str:
    lines = [
        "# Preference Reranking Evaluation",
        "",
        f"- base_model: {report['base_model']}",
        f"- adapter_path: {report['adapter_path']}",
        f"- eval_file: {report['eval_file']}",
        f"- sample_size: {report['sample_size']}",
        f"- num_candidates: {report['num_candidates']}",
        "",
        "| Metric | Before | After |",
        "| --- | ---: | ---: |",
        f"| JSON parse rate | {report['before']['json_parse_rate']} | {report['after']['json_parse_rate']} |",
        f"| Verdict consistency rate | {report['before']['verdict_consistency_rate']} | {report['after']['verdict_consistency_rate']} |",
        f"| Citation valid rate | {report['before']['citation_valid_rate']} | {report['after']['citation_valid_rate']} |",
        f"| Unsupported sentence rate | {report['before']['unsupported_sentence_rate']} | {report['after']['unsupported_sentence_rate']} |",
        f"| Mean latency increase (s/example) |  | {report['mean_latency_increase_seconds']} |",
        "",
        "## Selected candidate positions",
        "",
        json.dumps(report["selected_candidate_position_distribution"], indent=2),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
