"""Build reproducible FEVER and SciFact sample datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.sample_pipeline import (
    build_data_quality_markdown,
    build_fever_sample,
    build_quality_report,
    build_scifact_sample,
    write_sample_jsonl,
)
from data.build_evidence_corpus import build_evidence_corpus
from evaluation.reporting import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FEVER and SciFact sample artifacts.")
    parser.add_argument("--fever-train", type=int, default=200)
    parser.add_argument("--fever-val", type=int, default=50)
    parser.add_argument("--fever-test", type=int, default=50)
    parser.add_argument("--scifact-train", type=int, default=200)
    parser.add_argument("--scifact-val", type=int, default=50)
    parser.add_argument("--scifact-test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    fever_train, fever_val, fever_test = build_fever_sample(
        train_size=args.fever_train,
        val_size=args.fever_val,
        test_size=args.fever_test,
        seed=args.seed,
    )
    scifact_train, scifact_val, scifact_test, scifact_corpus = build_scifact_sample(
        train_size=args.scifact_train,
        val_size=args.scifact_val,
        test_size=args.scifact_test,
        seed=args.seed,
    )

    write_sample_jsonl(fever_train, output_dir / "fever_train.jsonl")
    write_sample_jsonl(fever_val, output_dir / "fever_val.jsonl")
    write_sample_jsonl(fever_test, output_dir / "fever_test.jsonl")
    write_sample_jsonl(scifact_train, output_dir / "scifact_train.jsonl")
    write_sample_jsonl(scifact_val, output_dir / "scifact_val.jsonl")
    write_sample_jsonl(scifact_test, output_dir / "scifact_test.jsonl")
    build_evidence_corpus(
        [*fever_train, *fever_val, *fever_test, *scifact_train, *scifact_val, *scifact_test],
        output_dir / "evidence_corpus.jsonl",
    )

    all_records = [*fever_train, *fever_val, *fever_test, *scifact_train, *scifact_val, *scifact_test]
    quality_report = build_quality_report(all_records)
    write_report(quality_report, reports_dir / "data_quality.json")
    (reports_dir / "data_quality.md").write_text(build_data_quality_markdown(quality_report), encoding="utf-8")

    print("Built sample datasets:")
    print(f"  FEVER records: {len(fever_train)} / {len(fever_val)} / {len(fever_test)}")
    print(f"  SciFact records: {len(scifact_train)} / {len(scifact_val)} / {len(scifact_test)}")
    print(f"  SciFact corpus passages: {len(scifact_corpus)}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
