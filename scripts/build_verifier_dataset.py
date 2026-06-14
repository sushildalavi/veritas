"""Build a clean claim/evidence verifier dataset from the large FEVER+SciFact sample.

Each output row has the schema::

    {"claim_id", "claim", "evidence", "label", "source", "evidence_type"}

``label`` is one of ``SUPPORTED`` / ``REFUTED`` / ``NOT_ENOUGH_INFO``.

``evidence_type`` is one of:
  - ``gold``: evidence text comes from the gold FEVER/SciFact annotation for a
    SUPPORTED/REFUTED claim.
  - ``no_evidence``: a NOT_ENOUGH_INFO example with empty evidence.
  - ``retrieved_negative``: a NOT_ENOUGH_INFO example paired with an unrelated
    gold evidence passage sampled from the same split, simulating retrieval
    that surfaces irrelevant evidence.

SUPPORTED/REFUTED records whose gold evidence has no usable text (after the
FEVER Wikipedia extraction fix) are skipped and counted in the audit report.

Outputs:
  data/processed/verifier_{train,val,test}.jsonl
  reports/verifier_data_audit.{json,md}
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from data.schemas import ClaimEvidenceRecord
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_records

SPLIT_SOURCES = {
    "train": ("fever_train", "scifact_train"),
    "val": ("fever_val", "scifact_val"),
    "test": ("fever_test", "scifact_test"),
}


def normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def gold_evidence_text(record: ClaimEvidenceRecord) -> str | None:
    texts = [span.text.strip() for span in record.evidence if span.text and span.text.strip()]
    if not texts:
        return None
    return " ".join(texts)


def build_examples(records: list[ClaimEvidenceRecord], *, seed: int) -> tuple[list[dict[str, Any]], int]:
    """Build verifier examples from a list of claim/evidence records.

    Returns the examples and the number of SUPPORTED/REFUTED records skipped
    because they had no usable gold evidence text.
    """

    rng = random.Random(seed)
    source = records[0].metadata.get("source", "unknown") if records else "unknown"

    gold_examples: list[dict[str, Any]] = []
    nei_records: list[ClaimEvidenceRecord] = []
    skipped = 0

    for record in records:
        label = normalize_label(record.label)
        if label == "NOT_ENOUGH_INFO":
            nei_records.append(record)
            continue
        evidence_text = gold_evidence_text(record)
        if evidence_text is None:
            skipped += 1
            continue
        gold_examples.append(
            {
                "claim_id": record.claim_id,
                "claim": record.claim,
                "evidence": evidence_text,
                "label": label,
                "source": record.metadata.get("source", source),
                "evidence_type": "gold",
            }
        )

    evidence_pool = [example["evidence"] for example in gold_examples]

    nei_examples: list[dict[str, Any]] = []
    for record in nei_records:
        use_retrieved_negative = bool(evidence_pool) and rng.random() < 0.5
        if use_retrieved_negative:
            evidence_text = rng.choice(evidence_pool)
            evidence_type = "retrieved_negative"
        else:
            evidence_text = ""
            evidence_type = "no_evidence"
        nei_examples.append(
            {
                "claim_id": record.claim_id,
                "claim": record.claim,
                "evidence": evidence_text,
                "label": "NOT_ENOUGH_INFO",
                "source": record.metadata.get("source", source),
                "evidence_type": evidence_type,
            }
        )

    return gold_examples + nei_examples, skipped


def write_jsonl(examples: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return output


def audit_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [example["claim"] for example in examples]
    duplicate_claims = len(claims) - len(set(claims))
    claim_lengths = [len(claim.split()) for claim in claims if claim]
    evidence_lengths = [len(example["evidence"].split()) for example in examples if example["evidence"]]
    return {
        "example_count": len(examples),
        "label_distribution": dict(Counter(example["label"] for example in examples)),
        "source_distribution": dict(Counter(example["source"] for example in examples)),
        "evidence_type_distribution": dict(Counter(example["evidence_type"] for example in examples)),
        "duplicate_claims": duplicate_claims,
        "missing_evidence_count": sum(1 for example in examples if not example["evidence"]),
        "average_claim_length": mean(claim_lengths) if claim_lengths else 0.0,
        "average_evidence_length": mean(evidence_lengths) if evidence_lengths else 0.0,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = ["# Verifier Data Audit", ""]
    lines.append("| split | examples | gold | no_evidence | retrieved_negative | skipped (no gold evidence) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for split_name in ("train", "val", "test"):
        split_report = report["splits"][split_name]
        evidence_types = split_report["evidence_type_distribution"]
        lines.append(
            f"| {split_name} | {split_report['example_count']} "
            f"| {evidence_types.get('gold', 0)} "
            f"| {evidence_types.get('no_evidence', 0)} "
            f"| {evidence_types.get('retrieved_negative', 0)} "
            f"| {split_report['skipped_missing_gold_evidence']} |"
        )
    lines.append("")
    for split_name in ("train", "val", "test"):
        split_report = report["splits"][split_name]
        lines.append(f"## {split_name}")
        lines.append("")
        lines.append(f"- label distribution: {split_report['label_distribution']}")
        lines.append(f"- source distribution: {split_report['source_distribution']}")
        lines.append(f"- duplicate claims: {split_report['duplicate_claims']}")
        lines.append(f"- average claim length (words): {split_report['average_claim_length']:.2f}")
        lines.append(f"- average evidence length (words): {split_report['average_evidence_length']:.2f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a clean verifier dataset from the large sample.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--suffix", default="_large")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_names = [name for sources in SPLIT_SOURCES.values() for name in sources]
    records_by_split = load_records(data_dir, split_names=split_names, suffix=args.suffix)

    report: dict[str, Any] = {"splits": {}}
    seen_triples: set[tuple[str, str, str]] = set()
    for split_index, (split_name, source_names) in enumerate(SPLIT_SOURCES.items()):
        records = [record for source_name in source_names for record in records_by_split.get(source_name, [])]
        examples, skipped = build_examples(records, seed=args.seed + split_index)

        # Drop examples whose (claim, evidence, label) triple already appeared in an
        # earlier split (train > val > test priority) to prevent cross-split leakage.
        cross_split_duplicates = 0
        deduped_examples: list[dict[str, Any]] = []
        for example in examples:
            triple = (example["claim"], example["evidence"], example["label"])
            if triple in seen_triples:
                cross_split_duplicates += 1
                continue
            deduped_examples.append(example)
            seen_triples.add(triple)
        examples = deduped_examples

        write_jsonl(examples, output_dir / f"verifier_{split_name}.jsonl")
        split_report = audit_examples(examples)
        split_report["skipped_missing_gold_evidence"] = skipped
        split_report["cross_split_duplicates_removed"] = cross_split_duplicates
        report["splits"][split_name] = split_report
        print(
            f"verifier_{split_name}: {len(examples)} examples "
            f"({skipped} skipped, no gold evidence; {cross_split_duplicates} cross-split duplicates removed)"
        )

    write_report(report, reports_dir / "verifier_data_audit.json")
    (reports_dir / "verifier_data_audit.md").write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {reports_dir}/verifier_data_audit.json and {reports_dir}/verifier_data_audit.md")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
