"""Train the Mac-local explanation-only MLX LoRA adapter."""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import yaml

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen2.5 MLX LoRA explanation adapter.")
    parser.add_argument("--config", default="configs/mlx_lora_explanation_qwen15b.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--data-dir", default="data/processed/explanation_sft")
    parser.add_argument("--eval-file", default="valid.jsonl")
    parser.add_argument("--max-eval-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--report-json", default="reports/mlx_lora_explanation_eval.json")
    parser.add_argument("--report-md", default="reports/mlx_lora_explanation_eval.md")
    parser.add_argument("--skip-train", action="store_true")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser as build_lora_parser, run as run_lora  # noqa: PLC0415

    config_path = Path(args.config)
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if args.iters is not None:
        config["iters"] = args.iters
    if args.model is not None:
        config["model"] = args.model
    if args.data is not None:
        config["data"] = args.data
    if args.adapter_path is not None:
        config["adapter_path"] = args.adapter_path

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
        print(f"Training explanation adapter for {base_model} -> {adapter_path}")
        run_lora(types.SimpleNamespace(**lora_args))

    print("Loading explanation adapter for evaluation")
    rows = read_jsonl(Path(args.data_dir) / args.eval_file)
    if args.max_eval_examples > 0:
        rows = rows[: args.max_eval_examples]
    if not rows:
        raise SystemExit(f"No evaluation examples found at {Path(args.data_dir) / args.eval_file}")

    from scripts.eval_mlx_lora_explanation import _build_generator  # noqa: PLC0415

    generator = _build_generator(base_model, str(adapter_path), max_new_tokens=args.max_new_tokens)
    report = _evaluate(rows, generator, base_model=base_model, adapter_path=str(adapter_path), eval_file=str(Path(args.data_dir) / args.eval_file))

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(f"mlx lora explanation eval: {report}")


def _evaluate(rows, generator, *, base_model: str, adapter_path: str, eval_file: str):
    from scripts.eval_mlx_lora_explanation import _extract_claim, _extract_evidence, _extract_verdict, _parse_json
    from rag.citation_checker import check_citations
    from rag.context_builder import ContextBundle
    from data.schemas import EvidenceSpan
    from time import perf_counter

    parsed = 0
    verdict_consistent = 0
    citation_valid = 0
    unsupported_sentence_rates: list[float] = []
    latencies: list[float] = []
    good_examples: list[dict[str, str]] = []
    failure_examples: list[dict[str, str]] = []

    for row in rows:
        started = perf_counter()
        response = generator(row)
        latencies.append(perf_counter() - started)
        payload = _parse_json(response)
        context = ContextBundle(
            claim=_extract_claim(row),
            evidence_items=(EvidenceSpan(doc_id="E1", text=_extract_evidence(row), title=None, score=None),),
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
    return {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "eval_file": eval_file,
        "sample_size": sample_size,
        "json_parse_rate": round(parsed / sample_size, 4),
        "verdict_consistency_rate": round(verdict_consistent / sample_size, 4),
        "citation_valid_rate": round(citation_valid / sample_size, 4),
        "unsupported_sentence_rate": round(sum(unsupported_sentence_rates) / sample_size, 4),
        "mean_latency_seconds": round(sum(latencies) / sample_size, 4),
        "good_examples": good_examples,
        "failure_examples": failure_examples,
    }


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
