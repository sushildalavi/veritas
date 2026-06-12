"""Shared loaders and summaries for sampled benchmark artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from data.schemas import ClaimEvidenceRecord, EvidenceSpan

SPLITS = ("fever_train", "fever_val", "fever_test", "scifact_train", "scifact_val", "scifact_test")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return [
        json.loads(line)
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_records(
    data_dir: str | Path,
    split_names: Iterable[str] = SPLITS,
    *,
    suffix: str = "",
) -> dict[str, list[ClaimEvidenceRecord]]:
    base = Path(data_dir)
    records: dict[str, list[ClaimEvidenceRecord]] = {}
    for split_name in split_names:
        split_path = base / f"{split_name}{suffix}.jsonl"
        records[split_name] = [
            ClaimEvidenceRecord(
                claim_id=str(row.get("claim_id", "")),
                claim=str(row.get("claim", "")),
                label=str(row.get("label", "NOT_ENOUGH_INFO")).replace("_", " "),
                evidence=tuple(
                    EvidenceSpan(
                        doc_id=str(item.get("doc_id", "")),
                        text=str(item.get("text", "")),
                        title=item.get("title"),
                        score=item.get("score"),
                        metadata=dict(item.get("metadata", {})),
                    )
                    for item in row.get("evidence", [])
                ),
                split=row.get("split"),
                metadata=dict(row.get("metadata", {})),
            )
            for row in read_jsonl(split_path)
        ]
    return records


def load_evidence_corpus(data_dir: str | Path, *, suffix: str = "") -> list[EvidenceSpan]:
    corpus_path = Path(data_dir) / f"evidence_corpus{suffix}.jsonl"
    return [
        EvidenceSpan(
            doc_id=str(row.get("doc_id", "")),
            text=str(row.get("text", "")),
            title=row.get("title"),
            score=row.get("score"),
            metadata=dict(row.get("metadata", {})),
        )
        for row in read_jsonl(corpus_path)
    ]


def iter_records(records: dict[str, list[ClaimEvidenceRecord]], prefixes: Iterable[str] | None = None) -> list[ClaimEvidenceRecord]:
    selected: list[ClaimEvidenceRecord] = []
    prefix_tuple = tuple(prefixes or records.keys())
    for split_name, split_records in records.items():
        if split_name.startswith(prefix_tuple):
            selected.extend(split_records)
    return selected


def relevant_doc_ids(record: ClaimEvidenceRecord) -> list[str]:
    return [span.doc_id for span in record.evidence]


def build_markdown_table(rows: list[dict[str, object]], headers: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"
