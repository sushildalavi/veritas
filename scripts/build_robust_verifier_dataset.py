"""Build a verifier robustness training set from retrieved-evidence pairs.

The production verifier (``checkpoints/transformer_verifier_clean``) is
trained only on oracle (claim, gold-evidence) pairs from
``data/processed/verifier_train.jsonl``. On the full 650-example v2 eval, its
per-passage macro-F1 drops from 0.6728 (oracle evidence) to 0.3887 (retrieved
evidence) -- the verifier rarely sees irrelevant or near-miss passages during
training, so it overpredicts SUPPORTED/REFUTED on weak retrieved evidence.

This script combines the existing oracle training set with the per-passage
retrieved-evidence calibration set
(``data/processed/retrieved_evidence_verifier_calibration.jsonl``, built by
``scripts/build_retrieved_evidence_dataset.py``), which already contains
positive, near-miss, same-topic, and irrelevant passages labeled
SUPPORTS/REFUTES/NEI. The combined set is written in the same schema as
``verifier_train.jsonl`` so it can be passed directly to
``scripts/train_transformer_verifier_clean.py --train-file ...``.

Writes:
  data/processed/verifier_train_robust.jsonl
  reports/verifier_train_robust_dataset.json / .md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl

PAIR_LABEL_TO_LABEL = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NEI": "NOT_ENOUGH_INFO",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a combined oracle + retrieved-evidence verifier training set.")
    parser.add_argument("--oracle-file", default="data/processed/verifier_train.jsonl")
    parser.add_argument("--retrieved-file", default="data/processed/retrieved_evidence_verifier_calibration.jsonl")
    parser.add_argument("--output", default="data/processed/verifier_train_robust.jsonl")
    parser.add_argument("--report-json", default="reports/verifier_train_robust_dataset.json")
    parser.add_argument("--report-md", default="reports/verifier_train_robust_dataset.md")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _convert_retrieved_pair(row: dict[str, object]) -> dict[str, object]:
    return {
        "claim_id": str(row.get("claim_id", "")),
        "claim": row.get("claim", ""),
        "evidence": row.get("passage_text", ""),
        "label": PAIR_LABEL_TO_LABEL[str(row["pair_label"])],
        "source": row.get("source", ""),
        "evidence_type": row.get("pair_type", "retrieved"),
    }


def _to_markdown(stats: dict[str, object]) -> str:
    lines = [
        "# Robust Verifier Training Dataset",
        "",
        f"- Oracle examples: {stats['oracle_count']}",
        f"- Retrieved-evidence examples: {stats['retrieved_count']}",
        f"- Total examples: {stats['total_count']}",
        "",
        "## Label distribution",
        "",
        "| label | oracle | retrieved | combined |",
        "| --- | --- | --- | --- |",
    ]
    for label in ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"):
        lines.append(
            f"| {label} | {stats['oracle_label_counts'].get(label, 0)} | "
            f"{stats['retrieved_label_counts'].get(label, 0)} | "
            f"{stats['combined_label_counts'].get(label, 0)} |"
        )
    lines.append("")
    lines.append("## Retrieved pair_type distribution")
    lines.append("")
    lines.append("| pair_type | count |")
    lines.append("| --- | --- |")
    for pair_type, count in sorted(stats["retrieved_pair_type_counts"].items()):
        lines.append(f"| {pair_type} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()

    oracle_examples = list(read_jsonl(Path(args.oracle_file)))
    retrieved_rows = list(read_jsonl(Path(args.retrieved_file)))
    retrieved_examples = [_convert_retrieved_pair(row) for row in retrieved_rows]

    combined = oracle_examples + retrieved_examples
    random.Random(args.seed).shuffle(combined)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in combined:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _label_counts(examples: list[dict[str, object]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for example in examples:
            label = str(example["label"])
            counts[label] = counts.get(label, 0) + 1
        return counts

    pair_type_counts: dict[str, int] = {}
    for row in retrieved_rows:
        pair_type = str(row.get("pair_type", "unknown"))
        pair_type_counts[pair_type] = pair_type_counts.get(pair_type, 0) + 1

    stats = {
        "oracle_file": args.oracle_file,
        "retrieved_file": args.retrieved_file,
        "output": str(output_path),
        "seed": args.seed,
        "oracle_count": len(oracle_examples),
        "retrieved_count": len(retrieved_examples),
        "total_count": len(combined),
        "oracle_label_counts": _label_counts(oracle_examples),
        "retrieved_label_counts": _label_counts(retrieved_examples),
        "combined_label_counts": _label_counts(combined),
        "retrieved_pair_type_counts": pair_type_counts,
    }

    write_report(stats, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(stats), encoding="utf-8")
    print(f"Wrote {len(combined)} examples to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
