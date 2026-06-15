"""Launch an OpenAI-compatible vLLM server for explanation generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_project_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch vLLM for Veritas explanation generation.")
    parser.add_argument("--config", default="configs/vllm_serving.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_vllm_command(
    *,
    model: str,
    host: str,
    port: int,
    api_key: str | None,
    max_model_len: int,
    tensor_parallel_size: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    if api_key:
        command.extend(["--api-key", api_key])
    return command


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings(args.config)
    command = build_vllm_command(
        model=args.model or settings.vllm_model,
        host=args.host,
        port=args.port,
        api_key=args.api_key or settings.vllm_api_key,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if args.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
