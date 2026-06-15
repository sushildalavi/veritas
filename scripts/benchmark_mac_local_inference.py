"""Benchmark Mac-local LLM inference backends, or write clear skipped reports.

This complements ``scripts/benchmark_inference_serving.py`` (transformers +
vLLM) with backends that run natively on Apple Silicon without CUDA:

- **mlx-lm**: measured if the ``mlx-lm`` package is installed (runs on the
  Metal/GPU backend via Apple's MLX framework).
- **ollama**: measured if an Ollama server is reachable at
  ``mac_local_backends.ollama.base_url`` (default
  ``http://localhost:11434``).
- **llama.cpp**: measured if both
  ``mac_local_backends.llama_cpp.binary_path`` and ``.model_path`` are
  configured and point to existing files (a llama.cpp binary built with
  Metal support, and a GGUF model).

Any backend that is unavailable is written as ``status: skipped`` with the
reason and exact setup steps -- no numbers are fabricated. vLLM/Triton remain
CUDA-only skipped reports (see ``scripts/benchmark_inference_serving.py`` /
``scripts/benchmark_triton_dense_scoring.py``); this script does not touch
those.

Writes ``reports/mac_local_inference_benchmark.json`` / ``.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from evaluation.reporting import write_report
from scripts.benchmark_verifier_runtime import _percentile
from scripts.eval_oracle_vs_retrieved_v2 import _git_commit_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Mac-local LLM inference backends (mlx-lm, ollama, llama.cpp).")
    parser.add_argument("--config", default="configs/inference_benchmark.yaml")
    parser.add_argument("--report-json", default="reports/mac_local_inference_benchmark.json")
    parser.add_argument("--report-md", default="reports/mac_local_inference_benchmark.md")
    return parser


def _benchmark_mlx_lm(config: dict[str, object], prompt: str) -> dict[str, object]:
    try:
        from mlx_lm import generate, load
    except ImportError:
        return {
            "status": "skipped",
            "backend": "mlx-lm",
            "reason": "the 'mlx-lm' package is not installed in this environment",
            "next_steps": "pip install mlx-lm, then re-run scripts/benchmark_mac_local_inference.py.",
        }

    model_name = config["model"]
    max_tokens = int(config["max_tokens"])
    num_iterations = int(config.get("num_iterations", 5))
    warmup_iterations = int(config.get("warmup_iterations", 1))

    model, tokenizer = load(model_name)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    for _ in range(warmup_iterations):
        generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max_tokens, verbose=False)

    latencies_ms = []
    token_counts = []
    for _ in range(num_iterations):
        started = perf_counter()
        output = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max_tokens, verbose=False)
        latencies_ms.append((perf_counter() - started) * 1000.0)
        token_counts.append(len(tokenizer.encode(output)))

    mean_latency_ms = mean(latencies_ms)
    mean_tokens = mean(token_counts)
    return {
        "status": "measured",
        "backend": "mlx-lm",
        "model": model_name,
        "device": "mps",
        "max_tokens": max_tokens,
        "iterations": num_iterations,
        "mean_latency_ms": round(mean_latency_ms, 2),
        "p50_latency_ms": round(_percentile(latencies_ms, 50), 2),
        "p95_latency_ms": round(_percentile(latencies_ms, 95), 2),
        "mean_generated_tokens": round(mean_tokens, 1),
        "tokens_per_sec": round(mean_tokens / (mean_latency_ms / 1000.0), 2),
        "requests_per_sec": round(1000.0 / mean_latency_ms, 3),
    }


def _ollama_healthcheck(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=2.0) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False


def _ollama_generate(base_url: str, model_name: str, prompt: str, max_tokens: int) -> tuple[float, int, float]:
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
    ).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = perf_counter()
    with urllib.request.urlopen(request, timeout=120.0) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))
    latency_ms = (perf_counter() - started) * 1000.0
    eval_count = int(body.get("eval_count", 0))
    eval_duration_ns = int(body.get("eval_duration", 0))
    tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0
    return latency_ms, eval_count, tokens_per_sec


def _benchmark_ollama(config: dict[str, object], prompt: str) -> dict[str, object]:
    base_url = str(config["base_url"])
    model_name = str(config["model"])

    if not _ollama_healthcheck(base_url):
        return {
            "status": "skipped",
            "backend": "ollama",
            "reason": f"ollama endpoint {base_url} is unreachable in this environment",
            "next_steps": (
                f"Start ollama (`ollama serve`), pull a model (`ollama pull {model_name}`), "
                "then re-run scripts/benchmark_mac_local_inference.py."
            ),
        }

    max_tokens = int(config["max_tokens"])
    num_iterations = int(config.get("num_iterations", 5))

    results = []
    error_count = 0
    timeout_count = 0
    for concurrency in config.get("concurrency_levels", [1]):
        num_requests = max(int(concurrency), num_iterations)
        latencies_ms: list[float] = []
        token_counts: list[int] = []
        tokens_per_sec_values: list[float] = []

        started = perf_counter()
        with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
            futures = [executor.submit(_ollama_generate, base_url, model_name, prompt, max_tokens) for _ in range(num_requests)]
            for future in futures:
                try:
                    latency_ms, eval_count, tps = future.result(timeout=120.0)
                    latencies_ms.append(latency_ms)
                    token_counts.append(eval_count)
                    tokens_per_sec_values.append(tps)
                except TimeoutError:
                    timeout_count += 1
                except (urllib.error.URLError, OSError, ValueError):
                    error_count += 1
        total_seconds = perf_counter() - started

        if latencies_ms:
            results.append(
                {
                    "concurrency": concurrency,
                    "num_requests": num_requests,
                    "mean_latency_ms": round(mean(latencies_ms), 2),
                    "p50_latency_ms": round(_percentile(latencies_ms, 50), 2),
                    "p95_latency_ms": round(_percentile(latencies_ms, 95), 2),
                    "mean_generated_tokens": round(mean(token_counts), 1),
                    "tokens_per_sec": round(mean(tokens_per_sec_values), 2),
                    "throughput_requests_per_sec": round(num_requests / total_seconds, 2),
                }
            )

    return {
        "status": "measured",
        "backend": "ollama",
        "model": model_name,
        "device": "mps (managed by ollama)",
        "base_url": base_url,
        "max_tokens": max_tokens,
        "results": results,
        "error_count": error_count,
        "timeout_count": timeout_count,
    }


def _parse_llama_cpp_tokens_per_sec(stderr_text: str) -> float | None:
    match = re.search(
        r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        stderr_text,
    )
    if match:
        return float(match.group(1))
    return None


def _benchmark_llama_cpp(config: dict[str, object], prompt: str) -> dict[str, object]:
    binary_path = config.get("binary_path")
    model_path = config.get("model_path")

    if not binary_path or not model_path:
        return {
            "status": "skipped",
            "backend": "llama.cpp",
            "reason": (
                "mac_local_backends.llama_cpp.binary_path / .model_path are not configured "
                "in configs/inference_benchmark.yaml"
            ),
            "next_steps": (
                "Build llama.cpp with Metal support (LLAMA_METAL=1), download a GGUF model, "
                "set mac_local_backends.llama_cpp.binary_path and .model_path in "
                "configs/inference_benchmark.yaml, then re-run scripts/benchmark_mac_local_inference.py."
            ),
        }

    binary = Path(str(binary_path))
    model_file = Path(str(model_path))
    if not binary.exists() or not model_file.exists():
        return {
            "status": "skipped",
            "backend": "llama.cpp",
            "reason": f"configured binary ({binary}) or model ({model_file}) does not exist",
            "next_steps": (
                "Verify mac_local_backends.llama_cpp.binary_path and .model_path point to "
                "existing files, then re-run scripts/benchmark_mac_local_inference.py."
            ),
        }

    max_tokens = int(config["max_tokens"])
    num_iterations = int(config.get("num_iterations", 5))

    latencies_ms = []
    tokens_per_sec_values = []
    for _ in range(num_iterations):
        started = perf_counter()
        proc = subprocess.run(
            [str(binary), "-m", str(model_file), "-p", prompt, "-n", str(max_tokens), "--no-display-prompt"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        latencies_ms.append((perf_counter() - started) * 1000.0)
        tps = _parse_llama_cpp_tokens_per_sec(proc.stderr)
        if tps is not None:
            tokens_per_sec_values.append(tps)

    mean_latency_ms = mean(latencies_ms)
    return {
        "status": "measured",
        "backend": "llama.cpp",
        "model": str(model_file),
        "device": "metal",
        "max_tokens": max_tokens,
        "iterations": num_iterations,
        "mean_latency_ms": round(mean_latency_ms, 2),
        "p50_latency_ms": round(_percentile(latencies_ms, 50), 2),
        "p95_latency_ms": round(_percentile(latencies_ms, 95), 2),
        "tokens_per_sec": round(mean(tokens_per_sec_values), 2) if tokens_per_sec_values else None,
        "requests_per_sec": round(1000.0 / mean_latency_ms, 3),
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Mac-Local Inference Benchmark",
        "",
        f"- Git commit: `{report.get('git_commit')}`",
        f"- Timestamp: {report.get('timestamp')}",
        "",
    ]

    for key, title in (("mlx_lm", "mlx-lm"), ("ollama", "Ollama"), ("llama_cpp", "llama.cpp")):
        backend = report[key]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- Status: {backend['status']}")
        if backend["status"] == "skipped":
            lines.append(f"- Reason: {backend['reason']}")
            lines.append(f"- Next steps: {backend['next_steps']}")
            lines.append("")
            continue

        lines.append(f"- Model: `{backend['model']}`")
        lines.append(f"- Device: {backend['device']}")
        lines.append(f"- max_tokens: {backend['max_tokens']}")
        lines.append("")

        if key == "ollama":
            lines.append("| concurrency | num_requests | mean_latency_ms | p50_latency_ms | p95_latency_ms | tokens/sec | requests/sec |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for row in backend["results"]:
                lines.append(
                    f"| {row['concurrency']} | {row['num_requests']} | {row['mean_latency_ms']} | "
                    f"{row['p50_latency_ms']} | {row['p95_latency_ms']} | {row['tokens_per_sec']} | "
                    f"{row['throughput_requests_per_sec']} |"
                )
            lines.append("")
            lines.append(f"- Error count: {backend['error_count']}")
            lines.append(f"- Timeout count: {backend['timeout_count']}")
        else:
            lines.append("| mean_latency_ms | p50_latency_ms | p95_latency_ms | tokens/sec | requests/sec |")
            lines.append("| --- | --- | --- | --- | --- |")
            lines.append(
                f"| {backend['mean_latency_ms']} | {backend['p50_latency_ms']} | {backend['p95_latency_ms']} | "
                f"{backend['tokens_per_sec']} | {backend['requests_per_sec']} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    backends_config = raw_config["mac_local_backends"]
    prompt = backends_config["prompt"]

    mlx_lm_report = _benchmark_mlx_lm(backends_config["mlx_lm"], prompt)
    ollama_report = _benchmark_ollama(backends_config["ollama"], prompt)
    llama_cpp_report = _benchmark_llama_cpp(backends_config["llama_cpp"], prompt)

    report = {
        "git_commit": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "mlx_lm": mlx_lm_report,
        "ollama": ollama_report,
        "llama_cpp": llama_cpp_report,
    }

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(
        f"mac_local_inference_benchmark complete: mlx_lm={mlx_lm_report['status']}, "
        f"ollama={ollama_report['status']}, llama_cpp={llama_cpp_report['status']}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
