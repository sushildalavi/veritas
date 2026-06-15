"""Benchmark the transformer verifier's inference latency and throughput.

Loads the local ``transformer_verifier_clean`` checkpoint with
``transformers``/``torch`` and measures forward-pass latency and throughput
across batch sizes on every available device (CPU, and MPS on Apple Silicon).
CUDA is included automatically if available; no device is faked.

Writes ``reports/verifier_inference_benchmark.json`` / ``.md``.
"""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.config import load_project_settings
from core.evidence_formatting import format_verifier_text
from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from scripts.eval_oracle_vs_retrieved_v2 import _git_commit_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark transformer verifier inference latency/throughput.")
    parser.add_argument("--config", default="configs/inference_benchmark.yaml")
    parser.add_argument("--report-json", default="reports/verifier_inference_benchmark.json")
    parser.add_argument("--report-md", default="reports/verifier_inference_benchmark.md")
    return parser


def _available_devices(requested: list[str]) -> list[str]:
    available = []
    for device in requested:
        if device == "cpu":
            available.append(device)
        elif device == "mps" and torch.backends.mps.is_available():
            available.append(device)
        elif device == "cuda" and torch.cuda.is_available():
            available.append(device)
    return available


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def _benchmark_device(
    *,
    model: torch.nn.Module,
    tokenizer,
    device: str,
    text: str,
    batch_sizes: list[int],
    num_iterations: int,
    warmup_iterations: int,
    max_length: int,
) -> list[dict[str, object]]:
    model = model.to(device)
    model.eval()
    results = []
    for batch_size in batch_sizes:
        batch_texts = [text] * batch_size
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            for _ in range(warmup_iterations):
                model(**inputs)
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()

            latencies_ms = []
            for _ in range(num_iterations):
                started = perf_counter()
                model(**inputs)
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()
                latencies_ms.append((perf_counter() - started) * 1000.0)

        mean_latency_ms = mean(latencies_ms)
        results.append(
            {
                "device": device,
                "batch_size": batch_size,
                "iterations": num_iterations,
                "mean_latency_ms": round(mean_latency_ms, 3),
                "p50_latency_ms": round(_percentile(latencies_ms, 50), 3),
                "p95_latency_ms": round(_percentile(latencies_ms, 95), 3),
                "throughput_examples_per_sec": round(batch_size / (mean_latency_ms / 1000.0), 2),
            }
        )
    return results


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Verifier Inference Benchmark",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Git commit: `{report.get('git_commit')}`",
        f"- torch version: `{report['torch_version']}`",
        f"- Devices tested: {', '.join(report['devices_tested'])}",
        "",
        "| device | batch_size | mean_latency_ms | p50_latency_ms | p95_latency_ms | throughput (examples/sec) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['device']} | {row['batch_size']} | {row['mean_latency_ms']} | {row['p50_latency_ms']} | "
            f"{row['p95_latency_ms']} | {row['throughput_examples_per_sec']} |"
        )
    lines.append("")
    lines.append(f"- Peak RSS: {report['peak_rss_mb']} MB")
    lines.append("")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings("configs/serving.yaml")

    import yaml

    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    verifier_config = raw_config["verifier"]

    checkpoint = verifier_config.get("checkpoint", settings.transformer_clean_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    text = format_verifier_text(
        verifier_config["sample_claim"],
        [EvidenceSpan(doc_id="bench", text=verifier_config["sample_passage"])],
    )

    devices = _available_devices(verifier_config.get("devices", ["cpu"]))
    all_results: list[dict[str, object]] = []
    for device in devices:
        all_results.extend(
            _benchmark_device(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text=text,
                batch_sizes=verifier_config.get("batch_sizes", [1]),
                num_iterations=verifier_config.get("num_iterations", 20),
                warmup_iterations=verifier_config.get("warmup_iterations", 3),
                max_length=verifier_config.get("max_length", 512),
            )
        )

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes, Linux in KB.
    peak_rss_mb = peak_rss_kb / (1024 * 1024) if sys.platform == "darwin" else peak_rss_kb / 1024

    report = {
        "checkpoint": str(checkpoint),
        "git_commit": _git_commit_hash(),
        "config_path": args.config,
        "torch_version": torch.__version__,
        "devices_tested": devices,
        "results": all_results,
        "peak_rss_mb": round(peak_rss_mb, 1),
    }

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(f"verifier_inference_benchmark complete: devices={devices}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
