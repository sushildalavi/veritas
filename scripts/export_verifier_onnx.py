"""Export the transformer verifier checkpoint to ONNX, or write a skipped report.

Attempts ``torch.onnx.export`` on ``checkpoints/transformer_verifier_clean``
(a RoBERTa sequence classifier). Exporting requires the ``onnx`` package; if
it is not installed, this script does NOT crash -- it writes
``reports/onnx_export.json`` / ``.md`` with ``status: skipped`` and the exact
command to install the missing dependency, then exits 0.

On success, writes the exported model to
``checkpoints/transformer_verifier_clean_onnx/model.onnx`` and a report with
``status: exported``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.config import load_project_settings
from evaluation.reporting import write_report
from scripts.eval_oracle_vs_retrieved_v2 import _git_commit_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the transformer verifier checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="checkpoints/transformer_verifier_clean_onnx/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--report-json", default="reports/onnx_export.json")
    parser.add_argument("--report-md", default="reports/onnx_export.md")
    return parser


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Verifier ONNX Export",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Git commit: `{report.get('git_commit')}`",
        f"- Status: {report['status']}",
    ]
    if report["status"] == "skipped":
        lines.extend(["", f"- Reason: {report['reason']}", f"- Next steps: {report['next_steps']}", ""])
    else:
        lines.extend(["", f"- Output: `{report['output']}`", f"- Opset: {report['opset']}", ""])
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings("configs/serving.yaml")
    checkpoint = args.checkpoint or settings.transformer_clean_checkpoint

    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "git_commit": _git_commit_hash(),
    }

    try:
        import onnx  # noqa: F401
    except ImportError:
        report["status"] = "skipped"
        report["reason"] = "the 'onnx' package is not installed in this environment"
        report["next_steps"] = "pip install onnx onnxruntime, then re-run scripts/export_verifier_onnx.py"
        write_report(report, Path(args.report_json))
        Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
        print("onnx_export skipped: 'onnx' package not installed")
        return

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()

    dummy_inputs = tokenizer("dummy input for export", return_tensors="pt")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
    )

    report["status"] = "exported"
    report["output"] = str(output_path)
    report["opset"] = args.opset
    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(f"onnx_export complete: {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
