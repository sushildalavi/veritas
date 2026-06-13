"""Build explanation-only SFT data for the Mac-local MLX LoRA path.

The dataset keeps verdict classification anchored to the verifier while
training the language model to emit faithful, structured explanations.

Outputs:
  data/processed/explanation_sft_train.jsonl
  data/processed/explanation_sft_val.jsonl
  data/processed/explanation_sft_test.jsonl
  data/processed/explanation_sft/{train,val,valid,test}.jsonl
  reports/explanation_sft_data_stats.{json,md}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Given a claim, evidence, and verifier "
    "verdict, generate a strict JSON object with keys verdict, explanation, "
    "and citations. The verdict must match the verifier verdict. The citation "
    "list must contain only E1."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build explanation-only SFT datasets.")
    parser.add_argument("--train-file", default="data/processed/verifier_train.jsonl")
    parser.add_argument("--val-file", default="data/processed/verifier_val.jsonl")
    parser.add_argument("--test-file", default="data/processed/verifier_test.jsonl")
    parser.add_argument("--output-prefix", default="data/processed/explanation_sft")
    parser.add_argument("--report-json", default="reports/explanation_sft_data_stats.json")
    parser.add_argument("--report-md", default="reports/explanation_sft_data_stats.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    splits = {
        "train": Path(args.train_file),
        "val": Path(args.val_file),
        "test": Path(args.test_file),
    }
    output_prefix = Path(args.output_prefix)
    output_prefix.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, Any]]] = {name: read_jsonl(path) for name, path in splits.items()}
    if not any(split_rows.values()):
        raise SystemExit("No verifier examples were found to build the explanation dataset.")

    stats = {
        "splits": {},
        "label_distribution": {},
        "source_distribution": {},
        "average_claim_length": {},
        "average_evidence_length": {},
        "output_prefix": str(output_prefix),
    }

    for split_name, rows in split_rows.items():
        if not rows:
            continue
        records = [_build_record(row) for row in rows]
        _write_jsonl(output_prefix.parent / f"{output_prefix.name}_{split_name}.jsonl", records)
        _write_jsonl(output_prefix / f"{split_name}.jsonl", records)
        if split_name == "val":
            _write_jsonl(output_prefix.parent / f"{output_prefix.name}_valid.jsonl", records)
            _write_jsonl(output_prefix / "valid.jsonl", records)
        stats["splits"][split_name] = len(records)
        label_counter = Counter(json.loads(record["messages"][2]["content"])["verdict"] for record in records)
        stats["label_distribution"][split_name] = dict(label_counter)
        source_counter = Counter(str(row.get("metadata", {}).get("source", "unknown")) for row in rows)
        stats["source_distribution"][split_name] = dict(source_counter)
        stats["average_claim_length"][split_name] = round(sum(len(str(row.get("claim", "")).split()) for row in rows) / len(rows), 2)
        stats["average_evidence_length"][split_name] = round(sum(len(str(row.get("evidence", "")).split()) for row in rows) / len(rows), 2)

    stats["total_examples"] = sum(stats["splits"].values())
    stats["prompt_template"] = "system/user/assistant JSON messages"
    stats["strict_json_output"] = True
    stats["citations"] = ["E1"]

    write_report(stats, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(stats), encoding="utf-8")
    print(f"Wrote explanation SFT data under {output_prefix}")


def _build_record(row: dict[str, Any]) -> dict[str, Any]:
    claim = str(row.get("claim", "")).strip()
    evidence = str(row.get("evidence", "")).strip()
    label = _normalize_label(row.get("label", "NOT_ENOUGH_INFO"))
    prompt = (
        f"Claim: {claim}\n\n"
        f"Evidence:\n[E1] {evidence}\n\n"
        f"Verifier verdict: {label}\n\n"
        "Task:\nGenerate a strict JSON object with keys verdict, explanation, and citations.\n"
        "Return only valid JSON."
    )
    assistant = json.dumps(
        {
            "verdict": label,
            "explanation": _build_explanation(claim, evidence, label),
            "citations": ["E1"],
        },
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ]
    }


def _build_explanation(claim: str, evidence: str, verdict: str) -> str:
    evidence_text = evidence.rstrip(".!?")
    if verdict == "NOT ENOUGH INFO":
        return f"The evidence does not establish the claim about {claim.lower()}."
    if not evidence_text:
        return f"The claim is {verdict.lower()} based on the evidence."
    return f"{evidence_text} therefore the claim is {verdict.lower()}."


def _normalize_label(raw: Any) -> str:
    text = str(raw).strip().upper().replace("-", "_")
    if text in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if text in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "# Explanation SFT Data Stats",
        "",
        f"- output_prefix: {stats['output_prefix']}",
        f"- total_examples: {stats['total_examples']}",
        f"- strict_json_output: {stats['strict_json_output']}",
        "",
        "| Split | Examples |",
        "| --- | ---: |",
    ]
    for split_name, count in stats["splits"].items():
        lines.append(f"| {split_name} | {count} |")
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
        "## Average lengths",
        "",
        json.dumps(
            {
                "average_claim_length": stats["average_claim_length"],
                "average_evidence_length": stats["average_evidence_length"],
            },
            indent=2,
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
