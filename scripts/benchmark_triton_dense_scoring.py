"""Optional Triton GPU microbenchmark for dense-retrieval scoring.

Dense retrieval scores a query embedding against a corpus of passage
embeddings via a matrix-vector dot product. This script benchmarks that
operation with a small Triton kernel against a plain PyTorch baseline
(``torch.matmul``) on CUDA.

Requires both the ``triton`` package and a CUDA device. If either is
unavailable (the default on this Apple Silicon development machine), writes
``reports/triton_dense_scoring_benchmark.json`` / ``.md`` with
``status: skipped`` and the exact environment required, then exits 0. No
CUDA/Triton results are fabricated.
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

import torch
import yaml

from evaluation.reporting import write_report
from scripts.benchmark_verifier_runtime import _percentile
from scripts.eval_oracle_vs_retrieved_v2 import _git_commit_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark dense-scoring with a Triton kernel vs torch.matmul.")
    parser.add_argument("--config", default="configs/inference_benchmark.yaml")
    parser.add_argument("--report-json", default="reports/triton_dense_scoring_benchmark.json")
    parser.add_argument("--report-md", default="reports/triton_dense_scoring_benchmark.md")
    return parser


def _to_markdown(report: dict[str, object]) -> str:
    lines = ["# Triton Dense-Scoring Microbenchmark", "", f"- Status: {report['status']}"]
    if report["status"] == "skipped":
        lines.extend(["", f"- Reason: {report['reason']}", f"- Next steps: {report['next_steps']}", ""])
        return "\n".join(lines)
    lines.extend(
        [
            f"- Git commit: `{report.get('git_commit')}`",
            f"- Vector dim: {report['vector_dim']}",
            "",
            "| batch_size | backend | mean_latency_ms | p50_latency_ms | p95_latency_ms |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["results"]:
        lines.append(
            f"| {row['batch_size']} | {row['backend']} | {row['mean_latency_ms']} | "
            f"{row['p50_latency_ms']} | {row['p95_latency_ms']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _run_benchmark(vector_dim: int, batch_sizes: list[int], num_iterations: int) -> list[dict[str, object]]:
    import triton
    import triton.language as tl

    @triton.jit
    def _dot_kernel(query_ptr, corpus_ptr, out_ptr, dim, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        query = tl.load(query_ptr + offsets, mask=mask, other=0.0)
        corpus_row = tl.load(corpus_ptr + row * dim + offsets, mask=mask, other=0.0)
        score = tl.sum(query * corpus_row, axis=0)
        tl.store(out_ptr + row, score)

    def triton_dense_scores(query: torch.Tensor, corpus: torch.Tensor) -> torch.Tensor:
        batch_size, dim = corpus.shape
        out = torch.empty(batch_size, device=corpus.device, dtype=corpus.dtype)
        block_size = triton.next_power_of_2(dim)
        _dot_kernel[(batch_size,)](query, corpus, out, dim, BLOCK_SIZE=block_size)
        return out

    results = []
    device = torch.device("cuda")
    for batch_size in batch_sizes:
        query = torch.randn(vector_dim, device=device)
        corpus = torch.randn(batch_size, vector_dim, device=device)

        for backend, fn in (
            ("torch_matmul", lambda: torch.matmul(corpus, query)),
            ("triton", lambda: triton_dense_scores(query, corpus)),
        ):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()

            latencies_ms = []
            for _ in range(num_iterations):
                started = perf_counter()
                fn()
                torch.cuda.synchronize()
                latencies_ms.append((perf_counter() - started) * 1000.0)

            mean_latency_ms = mean(latencies_ms)
            results.append(
                {
                    "batch_size": batch_size,
                    "backend": backend,
                    "iterations": num_iterations,
                    "mean_latency_ms": round(mean_latency_ms, 4),
                    "p50_latency_ms": round(_percentile(latencies_ms, 50), 4),
                    "p95_latency_ms": round(_percentile(latencies_ms, 95), 4),
                }
            )
    return results


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    triton_config = raw_config["triton_dense_scoring"]

    if not torch.cuda.is_available():
        report = {
            "status": "skipped",
            "reason": "no CUDA device available in this environment",
            "next_steps": (
                "Run on a host with an NVIDIA GPU and `pip install triton`, then re-run "
                "scripts/benchmark_triton_dense_scoring.py."
            ),
        }
        write_report(report, Path(args.report_json))
        Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
        print("triton_dense_scoring_benchmark skipped: no CUDA device available")
        return

    try:
        import triton  # noqa: F401
    except ImportError:
        report = {
            "status": "skipped",
            "reason": "the 'triton' package is not installed in this environment",
            "next_steps": "pip install triton, then re-run scripts/benchmark_triton_dense_scoring.py.",
        }
        write_report(report, Path(args.report_json))
        Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
        print("triton_dense_scoring_benchmark skipped: 'triton' package not installed")
        return

    results = _run_benchmark(
        vector_dim=triton_config["vector_dim"],
        batch_sizes=triton_config["batch_sizes"],
        num_iterations=triton_config.get("num_iterations", 20),
    )

    report = {
        "status": "measured",
        "git_commit": _git_commit_hash(),
        "vector_dim": triton_config["vector_dim"],
        "results": results,
    }
    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print("triton_dense_scoring_benchmark complete")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
