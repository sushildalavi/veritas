"""Convert the clean verifier dataset into MLX LoRA chat-format JSONL.

Reads ``data/processed/verifier_{train,val,test}.jsonl`` (claim, evidence,
label) and writes ``data/processed/mlx_lora/{train,valid,test}.jsonl`` in the
``{"messages": [...]}`` chat format expected by ``mlx_lm.lora --data``.

Each example asks the model to classify the claim and produce a
citation-grounded explanation using only the provided evidence, with the
assistant target built from ``rag.generate_template_explanation`` (the same
template explanation used elsewhere in this project).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from models.deberta_verifier import VerificationResult
from rag import build_context, generate_template_explanation

DISPLAY_LABEL = {"SUPPORTED": "SUPPORTED", "REFUTED": "REFUTED", "NOT_ENOUGH_INFO": "NOT ENOUGH INFO"}

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Given a claim and evidence, classify "
    "the claim as SUPPORTED, REFUTED, or NOT ENOUGH INFO and explain your "
    "verdict using only the provided evidence, citing evidence ids in "
    "brackets."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MLX LoRA chat-format dataset from the clean verifier dataset.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed/mlx_lora")
    parser.add_argument("--max-train-examples", type=int, default=600)
    parser.add_argument("--max-val-examples", type=int, default=100)
    parser.add_argument("--max-test-examples", type=int, default=100)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("verifier_train.jsonl", "train.jsonl", args.max_train_examples),
        ("verifier_val.jsonl", "valid.jsonl", args.max_val_examples),
        ("verifier_test.jsonl", "test.jsonl", args.max_test_examples),
    ]

    counts: dict[str, int] = {}
    for source_name, target_name, limit in splits:
        rows = read_jsonl(data_dir / source_name)
        if limit > 0:
            rows = rows[:limit]
        examples = [_build_example(row) for row in rows]
        _write_jsonl(output_dir / target_name, examples)
        counts[target_name] = len(examples)
        print(f"wrote {len(examples)} examples to {output_dir / target_name}")

    write_report({"output_dir": str(output_dir), "counts": counts}, output_dir / "manifest.json")


def _build_example(row: dict) -> dict:
    claim = str(row.get("claim", ""))
    evidence_text = str(row.get("evidence", ""))
    label = str(row.get("label", "NOT_ENOUGH_INFO"))

    context = build_context(claim, [EvidenceSpan(doc_id="1", text=evidence_text)] if evidence_text else [])
    verification = VerificationResult(verdict=DISPLAY_LABEL[label], confidence=1.0)
    template = generate_template_explanation(context, verification)

    evidence_block = context.format_block() or "(no evidence provided)"
    citation_text = " ".join(f"[{citation_id}]" for citation_id in template.citations) or "(none)"

    user_content = (
        f"Claim: {claim}\n"
        f"Evidence: {evidence_block}\n"
        "Classify the claim and explain using only the evidence."
    )
    assistant_content = (
        f"Verdict: {verification.verdict}\n"
        f"Explanation: {template.explanation}\n"
        f"Citation: {citation_text}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
