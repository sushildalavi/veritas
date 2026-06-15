"""Benchmark explanation serving backends or save a clear skipped report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_project_settings
from serving.model_loader import _build_explanation_generator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark explanation serving backends.")
    parser.add_argument("--config", default="configs/vllm_serving.yaml")
    parser.add_argument("--report-json", default="reports/vllm_benchmark.json")
    parser.add_argument("--report-md", default="reports/vllm_benchmark.md")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings(args.config)
    prompt = (
        'Return JSON with keys "explanation" and "citations". '
        "Claim: Paris is in France. Evidence: Paris is the capital of France."
    )
    report = {
        "config_path": args.config,
        "vllm_endpoint": settings.vllm_base_url,
        "vllm_model": settings.vllm_model,
        "backends": {},
    }

    template_settings = settings
    template_started = time.perf_counter()
    template_generator = _build_explanation_generator(template_settings)
    report["backends"]["configured_backend"] = {
        "mode": settings.explanation_mode,
        "available": template_generator is not None,
        "latency_ms": round((time.perf_counter() - template_started) * 1000.0, 3),
    }

    if not _vllm_healthcheck(settings.vllm_base_url):
        report["status"] = "skipped"
        report["reason"] = "vllm endpoint unavailable in current environment"
        _write_reports(report, Path(args.report_json), Path(args.report_md))
        print("vllm benchmark skipped: endpoint unavailable")
        return

    vllm_started = time.perf_counter()
    vllm_settings = load_project_settings(args.config)
    vllm_generator = _build_explanation_generator(vllm_settings)
    if vllm_generator is None:
        report["status"] = "skipped"
        report["reason"] = "vllm generator could not be constructed"
        _write_reports(report, Path(args.report_json), Path(args.report_md))
        print("vllm benchmark skipped: generator unavailable")
        return
    response = vllm_generator(prompt)
    report["status"] = "measured"
    report["backends"]["vllm"] = {
        "latency_ms": round((time.perf_counter() - vllm_started) * 1000.0, 3),
        "response_preview": response[:200],
    }
    _write_reports(report, Path(args.report_json), Path(args.report_md))
    print("vllm benchmark complete")


def _vllm_healthcheck(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=2.0) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _write_reports(report: dict[str, object], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# vLLM Benchmark",
        "",
        f"- Config path: `{report['config_path']}`",
        f"- vLLM endpoint: `{report['vllm_endpoint']}`",
        f"- vLLM model: `{report['vllm_model']}`",
        f"- Status: {report['status']}",
    ]
    if report["status"] == "skipped":
        lines.extend(["", f"- Reason: {report['reason']}", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "",
            "| backend | latency_ms | preview |",
            "| --- | --- | --- |",
            f"| vllm | {report['backends']['vllm']['latency_ms']} | {report['backends']['vllm']['response_preview']} |",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
