"""Compare MLX LoRA adapter eval reports and pick the best by verdict
accuracy, macro F1, citation validity, and unsupported-sentence rate.

Outputs:
  reports/mlx_lora_comparison.{json,md}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.reporting import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MLX LoRA adapter eval reports.")
    parser.add_argument(
        "--reports",
        nargs="+",
        default=["reports/mlx_lora_eval_200.json", "reports/mlx_lora_eval_300.json"],
        help="Eval report JSON files to compare, one per adapter.",
    )
    parser.add_argument(
        "--iters",
        nargs="+",
        type=int,
        default=[100, 300],
        help="LoRA iteration counts, one per --reports entry, for the comparison table.",
    )
    parser.add_argument("--output-json", default="reports/mlx_lora_comparison.json")
    parser.add_argument("--output-md", default="reports/mlx_lora_comparison.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if len(args.iters) != len(args.reports):
        raise SystemExit("--iters must have the same number of entries as --reports")

    entries = []
    for report_path, iters in zip(args.reports, args.iters):
        data = json.loads(Path(report_path).read_text(encoding="utf-8"))
        entries.append(
            {
                "report": report_path,
                "adapter_path": data["adapter_path"],
                "lora_iters": data.get("lora_iters", iters),
                "sample_size": data["sample_size"],
                "verdict_accuracy": data["verdict_accuracy"],
                "macro_f1": data["macro_f1"],
                "per_class_f1": data["per_class_f1"],
                "citation_valid_rate": data["citation_valid_rate"],
                "unsupported_sentence_rate": data["unsupported_sentence_rate"],
            }
        )

    best = max(entries, key=lambda entry: (entry["verdict_accuracy"], entry["macro_f1"], entry["citation_valid_rate"]))

    comparison = {"adapters": entries, "best_adapter": best["adapter_path"], "best_report": best["report"]}
    write_report(comparison, Path(args.output_json))
    Path(args.output_md).write_text(_to_markdown(comparison), encoding="utf-8")
    print(f"best adapter: {best['adapter_path']} ({best['report']})")


def _to_markdown(comparison: dict) -> str:
    lines = [
        "# MLX LoRA Adapter Comparison",
        "",
        "| Adapter | Iters | Sample size | Verdict acc | Macro F1 | Citation valid | Unsupported rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in comparison["adapters"]:
        lines.append(
            f"| {entry['adapter_path']} | {entry['lora_iters']} | {entry['sample_size']} | "
            f"{entry['verdict_accuracy']} | {entry['macro_f1']} | {entry['citation_valid_rate']} | "
            f"{entry['unsupported_sentence_rate']} |"
        )
    lines += ["", f"**Best adapter: `{comparison['best_adapter']}`** (`{comparison['best_report']}`)", ""]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
