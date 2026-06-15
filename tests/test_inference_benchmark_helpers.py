from scripts.benchmark_mac_local_inference import _parse_llama_cpp_tokens_per_sec, _to_markdown as mac_local_to_markdown
from scripts.benchmark_triton_dense_scoring import _to_markdown as triton_to_markdown
from scripts.benchmark_verifier_onnx import _to_markdown as onnx_bench_to_markdown
from scripts.benchmark_verifier_runtime import _available_devices, _percentile
from scripts.export_verifier_onnx import _to_markdown as onnx_export_to_markdown


def test_percentile_basic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert _percentile(values, 50) == 3.0
    assert _percentile(values, 0) == 1.0
    assert _percentile(values, 100) == 5.0


def test_available_devices_always_includes_cpu() -> None:
    devices = _available_devices(["cpu", "cuda"])

    assert "cpu" in devices
    assert "cuda" not in devices  # no CUDA on this machine


def test_onnx_export_skipped_markdown() -> None:
    report = {
        "checkpoint": "checkpoints/transformer_verifier_clean",
        "git_commit": "abc123",
        "status": "skipped",
        "reason": "the 'onnx' package is not installed in this environment",
        "next_steps": "pip install onnx onnxruntime",
    }

    markdown = onnx_export_to_markdown(report)

    assert "Status: skipped" in markdown
    assert "pip install onnx onnxruntime" in markdown


def test_onnx_verifier_benchmark_skipped_markdown() -> None:
    report = {
        "status": "skipped",
        "reason": "the 'onnxruntime' package is not installed in this environment",
        "next_steps": "pip install onnx onnxruntime",
    }

    markdown = onnx_bench_to_markdown(report)

    assert "Status: skipped" in markdown
    assert "onnxruntime" in markdown


def test_triton_benchmark_skipped_markdown() -> None:
    report = {
        "status": "skipped",
        "reason": "no CUDA device available in this environment",
        "next_steps": "Run on a host with an NVIDIA GPU and `pip install triton`.",
    }

    markdown = triton_to_markdown(report)

    assert "Status: skipped" in markdown
    assert "CUDA" in markdown


def test_triton_benchmark_measured_markdown() -> None:
    report = {
        "status": "measured",
        "git_commit": "abc123",
        "vector_dim": 384,
        "results": [
            {
                "batch_size": 32,
                "backend": "triton",
                "mean_latency_ms": 0.123,
                "p50_latency_ms": 0.12,
                "p95_latency_ms": 0.15,
            }
        ],
    }

    markdown = triton_to_markdown(report)

    assert "| 32 | triton | 0.123 | 0.12 | 0.15 |" in markdown


def test_parse_llama_cpp_tokens_per_sec() -> None:
    stderr_text = (
        "llama_print_timings:        eval time =   123.45 ms /    32 tokens "
        "(    3.86 ms per token,   259.21 tokens per second)"
    )

    assert _parse_llama_cpp_tokens_per_sec(stderr_text) == 259.21


def test_parse_llama_cpp_tokens_per_sec_no_match() -> None:
    assert _parse_llama_cpp_tokens_per_sec("no timing info here") is None


def test_mac_local_inference_skipped_markdown() -> None:
    report = {
        "git_commit": "abc123",
        "timestamp": "2026-06-15T00:00:00+00:00",
        "mlx_lm": {
            "status": "skipped",
            "reason": "the 'mlx-lm' package is not installed in this environment",
            "next_steps": "pip install mlx-lm",
        },
        "ollama": {
            "status": "skipped",
            "reason": "ollama endpoint http://localhost:11434 is unreachable in this environment",
            "next_steps": "Start ollama (`ollama serve`)",
        },
        "llama_cpp": {
            "status": "skipped",
            "reason": "mac_local_backends.llama_cpp.binary_path / .model_path are not configured",
            "next_steps": "Build llama.cpp with Metal support",
        },
    }

    markdown = mac_local_to_markdown(report)

    assert "## mlx-lm" in markdown
    assert "## Ollama" in markdown
    assert "## llama.cpp" in markdown
    assert markdown.count("Status: skipped") == 3
    assert "ollama endpoint http://localhost:11434 is unreachable" in markdown


def test_mac_local_inference_measured_markdown() -> None:
    report = {
        "git_commit": "abc123",
        "timestamp": "2026-06-15T00:00:00+00:00",
        "mlx_lm": {
            "status": "measured",
            "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "device": "mps",
            "max_tokens": 32,
            "mean_latency_ms": 595.85,
            "p50_latency_ms": 582.66,
            "p95_latency_ms": 621.71,
            "tokens_per_sec": 53.7,
            "requests_per_sec": 1.678,
        },
        "ollama": {
            "status": "skipped",
            "reason": "ollama endpoint http://localhost:11434 is unreachable in this environment",
            "next_steps": "Start ollama (`ollama serve`)",
        },
        "llama_cpp": {
            "status": "skipped",
            "reason": "mac_local_backends.llama_cpp.binary_path / .model_path are not configured",
            "next_steps": "Build llama.cpp with Metal support",
        },
    }

    markdown = mac_local_to_markdown(report)

    assert "| 595.85 | 582.66 | 621.71 | 53.7 | 1.678 |" in markdown
