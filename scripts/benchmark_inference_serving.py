"""Benchmark explanation-model serving: local transformers generation and vLLM.

Two backends are measured, each clearly marked as "measured" or "skipped":

- ``transformers``: a small local causal LM (TinyLlama-1.1B-Chat, already
  cached locally) is loaded with ``transformers``/``torch`` and benchmarked
  for single-request generation latency and tokens/sec on every available
  device (CPU, and MPS on Apple Silicon).
- ``vllm``: an OpenAI-compatible vLLM server is health-checked at the
  configured base URL. If reachable, concurrent request latency/throughput is
  measured via ``serving.vllm_client.VllmExplanationGenerator``. If not
  reachable, the section is written as ``status: skipped`` with the exact
  command needed to start a vLLM server (see docs/inference_performance.md).

Writes ``reports/inference_benchmark.json`` / ``.md``.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import load_project_settings
from evaluation.reporting import write_report
from scripts.benchmark_explanation_serving import _vllm_healthcheck
from scripts.benchmark_verifier_runtime import _available_devices
from scripts.eval_oracle_vs_retrieved_v2 import _git_commit_hash
from serving.vllm_client import VllmExplanationGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark explanation serving backends (transformers + vLLM).")
    parser.add_argument("--config", default="configs/inference_benchmark.yaml")
    parser.add_argument("--report-json", default="reports/inference_benchmark.json")
    parser.add_argument("--report-md", default="reports/inference_benchmark.md")
    return parser


def _benchmark_transformers_backend(config: dict[str, object]) -> dict[str, object]:
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.generation_config.max_length = None
    inputs_cpu = tokenizer(config["prompt"], return_tensors="pt")

    devices = _available_devices(config.get("devices", ["cpu"]))
    max_new_tokens = int(config["max_new_tokens"])
    num_iterations = int(config["num_iterations"])
    warmup_iterations = int(config["warmup_iterations"])

    results = []
    for device in devices:
        model = model.to(device)
        model.eval()
        inputs = {key: value.to(device) for key, value in inputs_cpu.items()}

        with torch.no_grad():
            for _ in range(warmup_iterations):
                model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()

            latencies_ms = []
            generated_token_counts = []
            for _ in range(num_iterations):
                started = perf_counter()
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()
                latencies_ms.append((perf_counter() - started) * 1000.0)
                generated_token_counts.append(output.shape[1] - inputs["input_ids"].shape[1])

        mean_latency_ms = mean(latencies_ms)
        mean_generated_tokens = mean(generated_token_counts)
        results.append(
            {
                "device": device,
                "iterations": num_iterations,
                "mean_latency_ms": round(mean_latency_ms, 2),
                "mean_generated_tokens": round(mean_generated_tokens, 1),
                "tokens_per_sec": round(mean_generated_tokens / (mean_latency_ms / 1000.0), 2),
            }
        )

    return {
        "status": "measured",
        "model_name": model_name,
        "devices_tested": devices,
        "max_new_tokens": max_new_tokens,
        "results": results,
    }


def _benchmark_vllm_backend(config: dict[str, object]) -> dict[str, object]:
    vllm_config_path = config["config_path"]
    settings = load_project_settings(vllm_config_path)

    if not _vllm_healthcheck(settings.vllm_base_url):
        return {
            "status": "skipped",
            "reason": "vllm endpoint unavailable in current environment",
            "vllm_base_url": settings.vllm_base_url,
            "vllm_model": settings.vllm_model,
            "next_steps": (
                "Start a vLLM server (see docs/vllm_serving.md / "
                "scripts/serve_vllm_explanations.py), then re-run "
                "scripts/benchmark_inference_serving.py."
            ),
        }

    generator = VllmExplanationGenerator(
        base_url=settings.vllm_base_url,
        model=settings.vllm_model,
        api_key=settings.vllm_api_key,
        timeout_seconds=settings.vllm_timeout_seconds,
    )
    prompt = (
        'Return JSON with keys "explanation" and "citations". '
        "Claim: Paris is in France. Evidence: Paris is the capital of France."
    )

    results = []
    for concurrency in config.get("concurrency_levels", [1]):
        num_requests = int(config.get("num_requests_per_level", concurrency))
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            list(executor.map(lambda _: generator(prompt), range(num_requests)))
        total_seconds = perf_counter() - started
        results.append(
            {
                "concurrency": concurrency,
                "num_requests": num_requests,
                "total_seconds": round(total_seconds, 3),
                "throughput_requests_per_sec": round(num_requests / total_seconds, 2),
            }
        )

    return {
        "status": "measured",
        "vllm_base_url": settings.vllm_base_url,
        "vllm_model": settings.vllm_model,
        "results": results,
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Inference Serving Benchmark",
        "",
        f"- Git commit: `{report.get('git_commit')}`",
        f"- torch version: `{report['torch_version']}`",
        "",
        "## transformers backend (local generation)",
        "",
        f"- Status: {report['transformers']['status']}",
        f"- Model: `{report['transformers']['model_name']}`",
        f"- Devices tested: {', '.join(report['transformers']['devices_tested'])}",
        f"- max_new_tokens: {report['transformers']['max_new_tokens']}",
        "",
        "| device | mean_latency_ms | mean_generated_tokens | tokens/sec |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["transformers"]["results"]:
        lines.append(f"| {row['device']} | {row['mean_latency_ms']} | {row['mean_generated_tokens']} | {row['tokens_per_sec']} |")

    lines.extend(["", "## vLLM backend", ""])
    vllm = report["vllm"]
    lines.append(f"- Status: {vllm['status']}")
    lines.append(f"- vLLM base URL: `{vllm['vllm_base_url']}`")
    lines.append(f"- vLLM model: `{vllm['vllm_model']}`")
    if vllm["status"] == "skipped":
        lines.append(f"- Reason: {vllm['reason']}")
        lines.append(f"- Next steps: {vllm['next_steps']}")
    else:
        lines.append("")
        lines.append("| concurrency | num_requests | total_seconds | throughput (requests/sec) |")
        lines.append("| --- | --- | --- | --- |")
        for row in vllm["results"]:
            lines.append(
                f"| {row['concurrency']} | {row['num_requests']} | {row['total_seconds']} | "
                f"{row['throughput_requests_per_sec']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    serving_config = raw_config["explanation_serving"]

    transformers_report = _benchmark_transformers_backend(serving_config["transformers_backend"])
    vllm_report = _benchmark_vllm_backend(serving_config["vllm"])

    report = {
        "git_commit": _git_commit_hash(),
        "config_path": args.config,
        "torch_version": torch.__version__,
        "transformers": transformers_report,
        "vllm": vllm_report,
    }

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(
        f"inference_benchmark complete: transformers={transformers_report['status']}, "
        f"vllm={vllm_report['status']}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
