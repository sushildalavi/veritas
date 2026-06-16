"""Build DPO preference data from the grounded explanation dataset.

The chosen response is the grounded completion from the SFT dataset.
The rejected response is a clearly synthetic failure mode, documented in
the metadata and report.
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

from data.explanation_artifacts import build_dpo_pair, validate_dpo_pair
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl

REJECTION_TYPES = (
    "wrong_label",
    "missing_citation",
    "hallucinated_fact",
    "vague_answer",
    "wrong_citation",
    "overclaiming",
    "insufficient_claim",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs for grounded explanations.")
    parser.add_argument("--train-file", default="data/explanations/sft_train.jsonl")
    parser.add_argument("--val-file", default="data/explanations/sft_val.jsonl")
    parser.add_argument("--output-train", default="data/explanations/dpo_train.jsonl")
    parser.add_argument("--output-val", default="data/explanations/dpo_val.jsonl")
    parser.add_argument("--report-json", default="reports/dpo_dataset_stats.json")
    parser.add_argument("--report-md", default="reports/dpo_dataset_stats.md")
    parser.add_argument("--samples-md", default="reports/dpo_preference_samples.md")
    parser.add_argument("--max-samples", type=int, default=10)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()

    train_rows = read_jsonl(Path(args.train_file))
    val_rows = read_jsonl(Path(args.val_file))
    if not train_rows or not val_rows:
        raise SystemExit("Missing SFT explanation data. Build the SFT dataset first.")

    train_pairs = _build_pairs(train_rows)
    val_pairs = _build_pairs(val_rows)
    for pair in train_pairs + val_pairs:
        validate_dpo_pair(pair)

    _write_jsonl(Path(args.output_train), train_pairs)
    _write_jsonl(Path(args.output_val), val_pairs)

    stats = _compute_stats(train_pairs, val_pairs, args)
    write_report(stats, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(stats), encoding="utf-8")
    Path(args.samples_md).write_text(_samples_markdown(train_pairs[: args.max_samples]), encoding="utf-8")
    print(f"Wrote DPO preference dataset to {args.output_train} and {args.output_val}")


def _build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        rejection_type = REJECTION_TYPES[index % len(REJECTION_TYPES)]
        pair = build_dpo_pair(row, rejection_type=rejection_type)
        pairs.append(pair)
    return pairs


def _compute_stats(train_pairs: list[dict[str, Any]], val_pairs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    all_pairs = train_pairs + val_pairs
    return {
        "output_train": str(args.output_train),
        "output_val": str(args.output_val),
        "train_examples": len(train_pairs),
        "val_examples": len(val_pairs),
        "total_examples": len(all_pairs),
        "label_distribution": dict(Counter(pair["verifier_label"] for pair in all_pairs)),
        "source_distribution": dict(Counter(str(pair.get("source", "unknown")) for pair in all_pairs)),
        "rejection_type_distribution": dict(Counter(pair["rejection_type"] for pair in all_pairs)),
        "synthetic_rejection": True,
        "prompt_source": "data/explanations/sft_{train,val}.jsonl",
    }


def _to_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "# DPO Dataset Stats",
        "",
        f"- output_train: {stats['output_train']}",
        f"- output_val: {stats['output_val']}",
        f"- total_examples: {stats['total_examples']}",
        f"- synthetic_rejection: {stats['synthetic_rejection']}",
        f"- prompt_source: {stats['prompt_source']}",
        "",
        "| Split | Examples |",
        "| --- | ---: |",
        f"| train | {stats['train_examples']} |",
        f"| val | {stats['val_examples']} |",
        "",
        "## Label distribution",
        "",
        json.dumps(stats["label_distribution"], indent=2),
        "",
        "## Source distribution",
        "",
        json.dumps(stats["source_distribution"], indent=2),
        "",
        "## Rejection type distribution",
        "",
        json.dumps(stats["rejection_type_distribution"], indent=2),
        "",
    ]
    return "\n".join(lines)


def _samples_markdown(samples: list[dict[str, Any]]) -> str:
    lines = [
        "# DPO Preference Samples",
        "",
        "Rejected responses are synthetic and intentionally contain one failure mode.",
        "",
    ]
    for sample in samples:
        lines += [
            f"## {sample['claim_id']} ({sample['rejection_type']})",
            "",
            "### Prompt",
            "",
            "```text",
            sample["prompt"],
            "```",
            "",
            "### Chosen",
            "",
            "```text",
            sample["chosen"],
            "```",
            "",
            "### Rejected",
            "",
            "```text",
            sample["rejected"],
            "```",
            "",
        ]
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
