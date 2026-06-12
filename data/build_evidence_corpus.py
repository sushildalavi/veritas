"""Build a JSONL evidence corpus from claim/evidence records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .schemas import ClaimEvidenceRecord, EvidenceSpan


def build_evidence_corpus(records: Iterable[ClaimEvidenceRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            for evidence in record.evidence:
                payload = {
                    "claim_id": record.claim_id,
                    "claim": record.claim,
                    "label": record.label,
                    "split": record.split,
                    "doc_id": evidence.doc_id,
                    "title": evidence.title,
                    "text": evidence.text,
                    "metadata": {
                        **record.metadata,
                        **evidence.metadata,
                    },
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return output
