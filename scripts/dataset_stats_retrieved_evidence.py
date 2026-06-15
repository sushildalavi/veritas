"""Compute summary statistics for the retrieved-evidence per-passage dataset.

Reads ``data/processed/retrieved_evidence_verifier_{calibration,holdout}.jsonl``
(built by ``scripts/build_retrieved_evidence_dataset.py``) and writes
``reports/retrieved_evidence_dataset_stats.json`` / ``.md`` with, per split:

- total examples (pairs)
- class distribution (SUPPORTS / REFUTES / NEI)
- FEVER vs SciFact distribution
- positive vs hard-negative counts, broken down by ``pair_type``
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl

SPLITS = ("calibration", "holdout")
POSITIVE_TYPES = {"positive_oracle", "positive_retrieved"}


def _split_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    pair_types = Counter(str(row["pair_type"]) for row in rows)
    positive = sum(count for ptype, count in pair_types.items() if ptype in POSITIVE_TYPES)
    hard_negative = sum(count for ptype, count in pair_types.items() if ptype not in POSITIVE_TYPES)
    return {
        "total_examples": len(rows),
        "claim_count": len({row["claim_id"] for row in rows}),
        "class_distribution": dict(Counter(str(row["pair_label"]) for row in rows)),
        "source_distribution": dict(Counter(str(row["source"]) for row in rows)),
        "positive_count": positive,
        "hard_negative_count": hard_negative,
        "pair_type_distribution": dict(pair_types),
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = ["# Retrieved-Evidence Per-Passage Dataset Stats", ""]
    lines.append("| split | total pairs | claims | SUPPORTS | REFUTES | NEI | fever | scifact | positive | hard_negative |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for split_name in SPLITS:
        s = report["splits"][split_name]
        cd, sd = s["class_distribution"], s["source_distribution"]
        lines.append(
            f"| {split_name} | {s['total_examples']} | {s['claim_count']} | {cd.get('SUPPORTS', 0)} | "
            f"{cd.get('REFUTES', 0)} | {cd.get('NEI', 0)} | {sd.get('fever', 0)} | {sd.get('scifact', 0)} | "
            f"{s['positive_count']} | {s['hard_negative_count']} |"
        )
    lines.append("")
    for split_name in SPLITS:
        s = report["splits"][split_name]
        lines.append(f"## {split_name}: pair_type distribution")
        lines.append("")
        for ptype, count in sorted(s["pair_type_distribution"].items()):
            lines.append(f"- {ptype}: {count}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entrypoint
    data_dir = Path("data/processed")
    report: dict[str, object] = {"splits": {}}
    for split_name in SPLITS:
        rows = read_jsonl(data_dir / f"retrieved_evidence_verifier_{split_name}.jsonl")
        report["splits"][split_name] = _split_stats(rows)

    write_report(report, Path("reports/retrieved_evidence_dataset_stats.json"))
    Path("reports/retrieved_evidence_dataset_stats.md").write_text(_to_markdown(report), encoding="utf-8")
    print("Wrote reports/retrieved_evidence_dataset_stats.json and .md")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
