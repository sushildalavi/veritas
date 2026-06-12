"""Build a meaningful, reproducible FEVER + SciFact sample for Veritas.

This replaces the toy 2/1/1 artifacts with a low-thousands-scale labelled
sample. SciFact is fully self-contained (claims + abstract corpus ship in one
archive), so it is used at full scale. FEVER claims/labels are cheap, but the
evidence text requires per-sentence Wikipedia fetches, so FEVER is bounded.

Outputs (prompt-named ``_large`` files in ``data/processed``):
  fever_{train,val,test}_large.jsonl
  scifact_{train,val,test}_large.jsonl
  evidence_corpus_large.jsonl
  reports/data_quality_large.{json,md}

If the downloads fail, a ``data_quality_large_FAILED.md`` report is written and
the script exits non-zero. No metrics are fabricated.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from data.build_evidence_corpus import build_evidence_corpus  # noqa: F401 (kept for parity)
from data.sample_pipeline import (
    build_data_quality_markdown,
    build_fever_sample,
    build_quality_report,
    build_scifact_sample,
    write_sample_jsonl,
)
from data.schemas import ClaimEvidenceRecord, EvidenceSpan
from evaluation.reporting import write_report

# ---------------------------------------------------------------------------
# Pure helpers (network-free, unit-tested with tiny fixtures).
# ---------------------------------------------------------------------------


def split_records(
    records: list[ClaimEvidenceRecord],
    *,
    val_size: int,
    seed: int,
) -> tuple[list[ClaimEvidenceRecord], list[ClaimEvidenceRecord]]:
    """Deterministically split a labelled pool into (val, test).

    SciFact's blind test set carries no labels, so we carve a held-out test
    split from the labelled dev pool instead.
    """

    pool = list(records)
    rng = random.Random(seed)
    rng.shuffle(pool)
    val_size = max(0, min(val_size, len(pool)))
    return pool[:val_size], pool[val_size:]


def corpus_rows_from_spans(spans: list[EvidenceSpan], *, source_split: str) -> list[dict[str, Any]]:
    """Convert evidence spans into deduplicated corpus rows keyed by doc_id."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for span in spans:
        doc_id = str(span.doc_id)
        if not doc_id or doc_id in seen:
            continue
        if not str(span.text).strip():
            continue
        seen.add(doc_id)
        rows.append(
            {
                "claim_id": "",
                "claim": "",
                "label": "",
                "split": source_split,
                "doc_id": doc_id,
                "title": span.title,
                "text": span.text,
                "metadata": dict(span.metadata),
            }
        )
    return rows


def assemble_evidence_corpus(
    scifact_corpus: list[EvidenceSpan],
    fever_records: list[ClaimEvidenceRecord],
) -> list[dict[str, Any]]:
    """Build a real retrieval haystack: full SciFact corpus + FEVER evidence.

    The full SciFact abstract corpus supplies genuine distractors so retrieval
    recall is non-trivial. FEVER evidence sentences are appended (deduped).
    """

    fever_spans = [span for record in fever_records for span in record.evidence]
    rows = corpus_rows_from_spans(list(scifact_corpus), source_split="scifact_corpus")
    seen = {row["doc_id"] for row in rows}
    for row in corpus_rows_from_spans(fever_spans, source_split="fever_evidence"):
        if row["doc_id"] not in seen:
            seen.add(row["doc_id"])
            rows.append(row)
    return rows


def write_corpus_rows(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


# ---------------------------------------------------------------------------
# Orchestration (network).
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the large FEVER + SciFact sample.")
    parser.add_argument("--fever-train", type=int, default=2000)
    parser.add_argument("--fever-val", type=int, default=500)
    parser.add_argument("--fever-test", type=int, default=500)
    parser.add_argument("--scifact-train", type=int, default=809)
    parser.add_argument("--scifact-dev", type=int, default=300)
    parser.add_argument("--scifact-val", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint (network)
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    try:
        fever_train, fever_val, fever_test = build_fever_sample(
            train_size=args.fever_train,
            val_size=args.fever_val,
            test_size=args.fever_test,
            seed=args.seed,
        )
        scifact_train, scifact_dev, _empty, scifact_corpus = build_scifact_sample(
            train_size=args.scifact_train,
            val_size=args.scifact_dev,
            test_size=0,
            seed=args.seed,
        )
    except Exception as exc:
        failure_path = reports_dir / "data_quality_large_FAILED.md"
        failure_path.write_text(
            "\n".join(
                [
                    "# Large Sample Build Failed",
                    "",
                    f"- fever sizes: {args.fever_train}/{args.fever_val}/{args.fever_test}",
                    f"- scifact sizes: {args.scifact_train}/{args.scifact_dev}",
                    "",
                    "The large sample build could not complete (download/parse error).",
                    "",
                    f"Reason: {type(exc).__name__}: {exc}",
                    "",
                    "No artifacts or metrics were fabricated.",
                ]
            ),
            encoding="utf-8",
        )
        raise SystemExit(f"Large sample build failed: {exc}") from exc

    scifact_val, scifact_test = split_records(scifact_dev, val_size=args.scifact_val, seed=args.seed)

    artifacts = {
        "fever_train": fever_train,
        "fever_val": fever_val,
        "fever_test": fever_test,
        "scifact_train": scifact_train,
        "scifact_val": scifact_val,
        "scifact_test": scifact_test,
    }
    for name, records in artifacts.items():
        write_sample_jsonl(records, output_dir / f"{name}_large.jsonl")

    corpus_rows = assemble_evidence_corpus(scifact_corpus, [*fever_train, *fever_val, *fever_test])
    write_corpus_rows(corpus_rows, output_dir / "evidence_corpus_large.jsonl")

    all_records = [record for records in artifacts.values() for record in records]
    quality_report = build_quality_report(all_records)
    quality_report["evidence_corpus_size"] = len(corpus_rows)
    quality_report["fever_sizes"] = {
        "train": len(fever_train),
        "val": len(fever_val),
        "test": len(fever_test),
    }
    quality_report["scifact_sizes"] = {
        "train": len(scifact_train),
        "val": len(scifact_val),
        "test": len(scifact_test),
    }
    quality_report["runtime_seconds"] = round(perf_counter() - started, 3)
    write_report(quality_report, reports_dir / "data_quality_large.json")
    (reports_dir / "data_quality_large.md").write_text(
        build_data_quality_markdown(quality_report)
        + "\n"
        + _scale_markdown(quality_report),
        encoding="utf-8",
    )

    print("Built large sample datasets:")
    for name, records in artifacts.items():
        print(f"  {name}_large: {len(records)}")
    print(f"  evidence_corpus_large: {len(corpus_rows)} passages")
    print(f"  runtime_seconds: {quality_report['runtime_seconds']}")


def _scale_markdown(report: dict[str, object]) -> str:
    lines = [
        "## Sample scale",
        "",
        f"- Evidence corpus passages: {report.get('evidence_corpus_size')}",
        f"- FEVER sizes (train/val/test): {report.get('fever_sizes')}",
        f"- SciFact sizes (train/val/test): {report.get('scifact_sizes')}",
        f"- Build runtime seconds: {report.get('runtime_seconds')}",
        "",
        "FEVER evidence text is fetched from Wikipedia and is bounded for runtime;",
        "SciFact ships claims + abstract corpus self-contained and is used at full scale.",
        "SciFact's blind test set carries no labels, so the held-out SciFact test split",
        "is carved deterministically from the labelled dev pool.",
    ]
    return "\n".join(lines) + "\n"


def _record_dump(record: ClaimEvidenceRecord) -> dict[str, Any]:  # pragma: no cover - debug helper
    payload = asdict(record)
    payload["evidence"] = [asdict(span) for span in record.evidence]
    return payload


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
