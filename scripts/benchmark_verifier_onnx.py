"""Benchmark the verifier via ONNX Runtime, or write a skipped report.

Requires both:
- the ``onnxruntime`` package, and
- an exported model at ``onnx_verifier.exported_model_path``
  (see ``configs/inference_benchmark.yaml`` and ``scripts/export_verifier_onnx.py``).

If either is missing, writes ``reports/onnx_verifier_benchmark.json`` / ``.md``
with ``status: skipped`` and the exact commands needed to produce them, then
exits 0. If both are present, measures forward-pass latency/throughput across
batch sizes, matching the PyTorch benchmark in
``scripts/benchmark_verifier_runtime.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from core.config import load_project_settings
from core.evidence_formatting import format_verifier_text
from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from scripts.benchmark_verifier_runtime import _percentile
from scripts.eval_oracle_vs_retrieved_v2 import _git_commit_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the verifier via ONNX Runtime.")
    parser.add_argument("--config", default="configs/inference_benchmark.yaml")
    parser.add_argument("--report-json", default="reports/onnx_verifier_benchmark.json")
    parser.add_argument("--report-md", default="reports/onnx_verifier_benchmark.md")
    return parser


def _to_markdown(report: dict[str, object]) -> str:
    lines = ["# Verifier ONNX Runtime Benchmark", "", f"- Status: {report['status']}"]
    if report["status"] == "skipped":
        lines.extend(["", f"- Reason: {report['reason']}", f"- Next steps: {report['next_steps']}", ""])
        return "\n".join(lines)
    lines.extend(
        [
            f"- Checkpoint: `{report['checkpoint']}`",
            f"- Git commit: `{report.get('git_commit')}`",
            "",
            "| batch_size | mean_latency_ms | p50_latency_ms | p95_latency_ms | throughput (examples/sec) |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["results"]:
        lines.append(
            f"| {row['batch_size']} | {row['mean_latency_ms']} | {row['p50_latency_ms']} | "
            f"{row['p95_latency_ms']} | {row['throughput_examples_per_sec']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    onnx_config = raw_config["onnx_verifier"]

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        report = {
            "status": "skipped",
            "reason": "the 'onnxruntime' package is not installed in this environment",
            "next_steps": (
                "pip install onnx onnxruntime, then run scripts/export_verifier_onnx.py "
                "and re-run this benchmark."
            ),
        }
        write_report(report, Path(args.report_json))
        Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
        print("onnx_verifier_benchmark skipped: 'onnxruntime' package not installed")
        return

    model_path = Path(onnx_config["exported_model_path"])
    if not model_path.exists():
        report = {
            "status": "skipped",
            "reason": f"no exported ONNX model found at {model_path}",
            "next_steps": "pip install onnx onnxruntime, then run scripts/export_verifier_onnx.py and re-run this benchmark.",
        }
        write_report(report, Path(args.report_json))
        Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
        print(f"onnx_verifier_benchmark skipped: no exported model at {model_path}")
        return

    from serving.onnx_verifier_backend import ONNXVerifierBackend

    settings = load_project_settings("configs/serving.yaml")
    checkpoint = onnx_config.get("checkpoint", settings.transformer_clean_checkpoint)
    backend = ONNXVerifierBackend(model_path=model_path, tokenizer_path=checkpoint)

    verifier_config = raw_config["verifier"]
    text = format_verifier_text(
        verifier_config["sample_claim"],
        [EvidenceSpan(doc_id="bench", text=verifier_config["sample_passage"])],
    )

    num_iterations = onnx_config.get("num_iterations", 20)
    warmup_iterations = onnx_config.get("warmup_iterations", 3)
    results = []
    for batch_size in onnx_config.get("batch_sizes", [1]):
        texts = [text] * batch_size
        for _ in range(warmup_iterations):
            backend.predict_logits(texts)

        latencies_ms = []
        for _ in range(num_iterations):
            started = perf_counter()
            backend.predict_logits(texts)
            latencies_ms.append((perf_counter() - started) * 1000.0)

        mean_latency_ms = mean(latencies_ms)
        results.append(
            {
                "batch_size": batch_size,
                "iterations": num_iterations,
                "mean_latency_ms": round(mean_latency_ms, 3),
                "p50_latency_ms": round(_percentile(latencies_ms, 50), 3),
                "p95_latency_ms": round(_percentile(latencies_ms, 95), 3),
                "throughput_examples_per_sec": round(batch_size / (mean_latency_ms / 1000.0), 2),
            }
        )

    report = {
        "status": "measured",
        "checkpoint": str(checkpoint),
        "git_commit": _git_commit_hash(),
        "model_path": str(model_path),
        "results": results,
    }
    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print("onnx_verifier_benchmark complete")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
