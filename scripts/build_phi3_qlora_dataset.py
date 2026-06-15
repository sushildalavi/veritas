"""Build Phi-3 QLoRA chat-format data from the clean verifier dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from models.deberta_verifier import VerificationResult
from rag import build_context, generate_template_explanation

DISPLAY_LABEL = {"SUPPORTED": "SUPPORTED", "REFUTED": "REFUTED", "NOT_ENOUGH_INFO": "NOT ENOUGH INFO"}
SYSTEM_PROMPT = (
    "You are Veritas, an evidence-grounded fact verification assistant. "
    "Use only the provided evidence. Return a verdict, concise explanation, and citations."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phi-3 QLoRA dataset.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed/phi3_qlora")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for source_name, target_name in (
        ("verifier_train.jsonl", "train.jsonl"),
        ("verifier_val.jsonl", "valid.jsonl"),
        ("verifier_test.jsonl", "test.jsonl"),
    ):
        rows = read_jsonl(Path(args.data_dir) / source_name)
        rendered = [_build_row(row) for row in rows]
        _write_jsonl(output_dir / target_name, rendered)
        manifest[target_name] = len(rendered)
    write_report({"counts": manifest, "output_dir": str(output_dir)}, output_dir / "manifest.json")
    print(f"wrote phi3 qlora dataset to {output_dir}")


def _build_row(row: dict) -> dict:
    claim = str(row.get("claim", ""))
    evidence_text = str(row.get("evidence", ""))
    label = str(row.get("label", "NOT_ENOUGH_INFO"))
    context = build_context(claim, [EvidenceSpan(doc_id="1", text=evidence_text)] if evidence_text else [])
    verification = VerificationResult(verdict=DISPLAY_LABEL[label], confidence=1.0)
    explanation = generate_template_explanation(context, verification)
    user_content = (
        f"Claim: {claim}\n"
        f"Evidence: {context.format_block() or '(no evidence provided)'}\n"
        "Respond with JSON keys verdict, explanation, and citations."
    )
    assistant_payload = {
        "verdict": verification.verdict,
        "explanation": explanation.explanation,
        "citations": explanation.citations,
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
        ]
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
