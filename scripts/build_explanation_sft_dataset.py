"""Build grounded explanation SFT data.

Inputs:
  data/processed/verifier_{train,val,test}.jsonl

Outputs:
  data/explanations/sft_train.jsonl
  data/explanations/sft_val.jsonl
  data/explanations/sft_test.jsonl
  reports/explanation_sft_dataset_stats.json
  reports/explanation_sft_dataset_stats.md
  reports/explanation_sft_samples.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.explanation_artifacts import build_explanation_record, validate_explanation_record
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build grounded explanation SFT datasets.")
    parser.add_argument("--train-file", default="data/processed/verifier_train.jsonl")
    parser.add_argument("--val-file", default="data/processed/verifier_val.jsonl")
    parser.add_argument("--test-file", default="data/processed/verifier_test.jsonl")
    parser.add_argument("--output-dir", default="data/explanations")
    parser.add_argument("--report-json", default="reports/explanation_sft_dataset_stats.json")
    parser.add_argument("--report-md", default="reports/explanation_sft_dataset_stats.md")
    parser.add_argument("--samples-md", default="reports/explanation_sft_samples.md")
    parser.add_argument("--max-samples", type=int, default=10)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()

    split_paths = {
        "train": Path(args.train_file),
        "val": Path(args.val_file),
        "test": Path(args.test_file),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "output_dir": str(output_dir),
        "source_files": {split: str(path) for split, path in split_paths.items()},
        "splits": {},
        "label_distribution": {},
        "source_distribution": {},
        "empty_evidence_count": {},
        "total_examples": 0,
        "prompt_version": "v1",
        "completion_version": "v1",
    }
    sample_rows: list[dict[str, Any]] = []

    for split, input_path in split_paths.items():
        rows = read_jsonl(input_path)
        records = [build_explanation_record(row, split=split) for row in rows]
        for record in records:
            validate_explanation_record(record)

        _write_jsonl(output_dir / f"sft_{split}.jsonl", records)

        stats["splits"][split] = len(records)
        stats["total_examples"] += len(records)
        stats["label_distribution"][split] = dict(Counter(record["verifier_label"] for record in records))
        stats["source_distribution"][split] = dict(Counter(str(record.get("source", "unknown")) for record in records))
        stats["empty_evidence_count"][split] = sum(1 for record in records if not record["evidence_passages"])

        if not sample_rows:
            sample_rows.extend(records[: args.max_samples])

    label_totals = Counter()
    source_totals = Counter()
    for split_labels in stats["label_distribution"].values():
        label_totals.update(split_labels)
    for split_sources in stats["source_distribution"].values():
        source_totals.update(split_sources)
    stats["label_totals"] = dict(label_totals)
    stats["source_totals"] = dict(source_totals)
    stats["sample_count"] = len(sample_rows)
    stats["sample_ids"] = [row["claim_id"] for row in sample_rows]
    stats["has_supported"] = stats["label_totals"].get("SUPPORTED", 0) > 0
    stats["has_refuted"] = stats["label_totals"].get("REFUTED", 0) > 0
    stats["has_nei"] = stats["label_totals"].get("NOT_ENOUGH_INFO", 0) > 0
    stats["has_fever"] = stats["source_totals"].get("fever", 0) > 0
    stats["has_scifact"] = stats["source_totals"].get("scifact", 0) > 0

    write_report(stats, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(stats), encoding="utf-8")
    Path(args.samples_md).write_text(_samples_markdown(sample_rows), encoding="utf-8")
    print(f"Wrote grounded explanation SFT dataset to {output_dir}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "# Explanation SFT Dataset Stats",
        "",
        f"- output_dir: {stats['output_dir']}",
        f"- total_examples: {stats['total_examples']}",
        f"- prompt_version: {stats['prompt_version']}",
        f"- completion_version: {stats['completion_version']}",
        "",
        "## Split sizes",
        "",
        "| Split | Examples | Empty evidence |",
        "| --- | ---: | ---: |",
    ]
    for split, count in stats["splits"].items():
        lines.append(f"| {split} | {count} | {stats['empty_evidence_count'].get(split, 0)} |")

    lines += [
        "",
        "## Label distribution",
        "",
        json.dumps(stats["label_distribution"], indent=2),
        "",
        "## Source distribution",
        "",
        json.dumps(stats["source_distribution"], indent=2),
        "",
        "## Coverage checks",
        "",
        f"- supported: {stats['has_supported']}",
        f"- refuted: {stats['has_refuted']}",
        f"- not_enough_info: {stats['has_nei']}",
        f"- fever: {stats['has_fever']}",
        f"- scifact: {stats['has_scifact']}",
        "",
    ]
    return "\n".join(lines)


def _samples_markdown(samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Explanation SFT Samples",
        "",
        "The examples below are direct artifacts from the grounded explanation dataset.",
        "",
    ]
    for sample in samples:
        lines += [
            f"## {sample['claim_id']} ({sample['source']}, {sample['verifier_label']})",
            "",
            "### Prompt",
            "",
            "```text",
            sample["prompt"],
            "```",
            "",
            "### Completion",
            "",
            "```text",
            sample["completion"],
            "```",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
