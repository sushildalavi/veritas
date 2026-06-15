"""Build DPO preference pairs from the clean verifier training set.

Reads ``data/processed/verifier_train.jsonl`` (claim, evidence, label,
source) and, for each example with non-empty evidence, builds a
``{"prompt", "chosen", "rejected"}`` pair in the format expected by TRL's
``DPOTrainer``:

- ``chosen``: the correct verdict with a grounded, citation-valid
  explanation built from ``rag.generate_template_explanation`` (the same
  template used by ``scripts/build_mlx_lora_dataset.py``).
- ``rejected``: one of three deterministic failure modes, rotated by row
  index so the dataset covers all three:
    - ``wrong_verdict``: correct citation/explanation shape, but the wrong
      verdict.
    - ``invalid_citation``: correct verdict, but cites an evidence id that
      does not exist in the context.
    - ``unsupported_explanation``: correct verdict, but a generic
      explanation with no citation at all.

Outputs:
  data/processed/preference_pairs.jsonl
  reports/preference_pair_stats.{json,md}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.schemas import EvidenceSpan
from evaluation.mlx_lora_eval import DISPLAY_LABEL, SYSTEM_PROMPT
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from models.deberta_verifier import VerificationResult
from rag import build_context, check_citations, generate_template_explanation

WRONG_VERDICT = {
    "SUPPORTED": "REFUTED",
    "REFUTED": "SUPPORTED",
    "NOT ENOUGH INFO": "SUPPORTED",
}
REJECTION_TYPES = ["wrong_verdict", "invalid_citation", "unsupported_explanation"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs from the clean verifier training set.")
    parser.add_argument("--input-file", default="data/processed/verifier_train.jsonl")
    parser.add_argument("--max-examples", type=int, default=0, help="0 = use all examples with non-empty evidence.")
    parser.add_argument("--output-jsonl", default="data/processed/preference_pairs.jsonl")
    parser.add_argument("--output-json", default="reports/preference_pair_stats.json")
    parser.add_argument("--output-md", default="reports/preference_pair_stats.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rows = read_jsonl(Path(args.input_file))
    rows = [row for row in rows if str(row.get("evidence", "")).strip()]
    if not rows:
        raise SystemExit(f"No examples with non-empty evidence found in {args.input_file}")

    pairs: list[dict] = []
    for row in rows:
        pair = _build_pair(row, REJECTION_TYPES[len(pairs) % len(REJECTION_TYPES)])
        if not pair["chosen_citation_valid"]:
            continue
        pairs.append(pair)
        if args.max_examples > 0 and len(pairs) >= args.max_examples:
            break

    if not pairs:
        raise SystemExit("No examples produced a citation-valid chosen response")

    _validate_pairs(pairs)
    stats = _compute_stats(pairs, input_file=args.input_file, output_jsonl=args.output_jsonl)
    _write_jsonl(Path(args.output_jsonl), [_strip_internal_fields(pair) for pair in pairs])
    write_report(stats, Path(args.output_json))
    Path(args.output_md).write_text(_to_markdown(stats), encoding="utf-8")
    print(f"wrote {len(pairs)} preference pairs to {args.output_jsonl}")


def _build_pair(row: dict, rejection_type: str) -> dict:
    claim = str(row.get("claim", ""))
    evidence_text = str(row.get("evidence", ""))
    label = str(row.get("label", "NOT_ENOUGH_INFO"))
    gold_verdict = DISPLAY_LABEL[label]

    context = build_context(claim, [EvidenceSpan(doc_id="1", text=evidence_text)])
    evidence_block = context.format_block()
    user_content = (
        f"Claim: {claim}\n"
        f"Evidence: {evidence_block}\n"
        "Classify the claim and explain using only the evidence."
    )
    prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"

    chosen_template = generate_template_explanation(context, VerificationResult(verdict=gold_verdict, confidence=1.0))
    chosen_citation = " ".join(f"[{citation_id}]" for citation_id in chosen_template.citations) or "(none)"
    chosen = f"Verdict: {gold_verdict}\nExplanation: {chosen_template.explanation}\nCitation: {chosen_citation}"
    chosen_citation_valid = check_citations(chosen_template.explanation, context).valid

    if rejection_type == "wrong_verdict":
        wrong_verdict = WRONG_VERDICT[gold_verdict]
        wrong_template = generate_template_explanation(context, VerificationResult(verdict=wrong_verdict, confidence=1.0))
        wrong_citation = " ".join(f"[{citation_id}]" for citation_id in wrong_template.citations) or "(none)"
        rejected = f"Verdict: {wrong_verdict}\nExplanation: {wrong_template.explanation}\nCitation: {wrong_citation}"
    elif rejection_type == "invalid_citation":
        invalid_id = max((item.citation_id for item in context.evidence_items), default=0) + 1
        evidence_stub = evidence_text.rstrip(".!?")
        rejected_explanation = f"[{invalid_id}] {evidence_stub} therefore the claim is {gold_verdict.lower()}."
        rejected = f"Verdict: {gold_verdict}\nExplanation: {rejected_explanation}\nCitation: [{invalid_id}]"
    else:  # unsupported_explanation
        rejected_explanation = f"The claim is {gold_verdict.lower()} based on general background knowledge."
        rejected = f"Verdict: {gold_verdict}\nExplanation: {rejected_explanation}\nCitation: (none)"

    return {
        "claim_id": row.get("claim_id"),
        "source": row.get("source"),
        "gold_label": gold_verdict,
        "rejection_type": rejection_type,
        "chosen_citation_valid": chosen_citation_valid,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def _strip_internal_fields(pair: dict) -> dict:
    return {key: value for key, value in pair.items() if key != "chosen_citation_valid"}


def _validate_pairs(pairs: list[dict]) -> None:
    for pair in pairs:
        for key in ("prompt", "chosen", "rejected"):
            if not pair[key].strip():
                raise ValueError(f"empty {key} for claim_id={pair.get('claim_id')}")
        if pair["chosen"] == pair["rejected"]:
            raise ValueError(f"chosen == rejected for claim_id={pair.get('claim_id')}")


def _compute_stats(pairs: list[dict], *, input_file: str, output_jsonl: str) -> dict:
    label_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    chosen_valid = 0
    for pair in pairs:
        label_counts[pair["gold_label"]] = label_counts.get(pair["gold_label"], 0) + 1
        source = str(pair["source"])
        source_counts[source] = source_counts.get(source, 0) + 1
        rejection_counts[pair["rejection_type"]] = rejection_counts.get(pair["rejection_type"], 0) + 1
        if pair["chosen_citation_valid"]:
            chosen_valid += 1

    sample_size = len(pairs)

    def avg_words(field: str) -> float:
        return round(sum(len(pair[field].split()) for pair in pairs) / sample_size, 2)

    return {
        "input_file": input_file,
        "output_jsonl": output_jsonl,
        "sample_size": sample_size,
        "label_distribution": label_counts,
        "source_distribution": source_counts,
        "rejection_type_distribution": rejection_counts,
        "chosen_citation_valid_count": chosen_valid,
        "chosen_citation_valid_rate": round(chosen_valid / sample_size, 4),
        "avg_prompt_words": avg_words("prompt"),
        "avg_chosen_words": avg_words("chosen"),
        "avg_rejected_words": avg_words("rejected"),
    }


def _to_markdown(stats: dict) -> str:
    lines = [
        "# DPO Preference Pair Stats",
        "",
        f"- input_file: {stats['input_file']}",
        f"- output_jsonl: {stats['output_jsonl']}",
        f"- sample_size: {stats['sample_size']}",
        f"- chosen_citation_valid_rate: {stats['chosen_citation_valid_rate']} "
        f"({stats['chosen_citation_valid_count']}/{stats['sample_size']})",
        f"- avg_prompt_words: {stats['avg_prompt_words']}",
        f"- avg_chosen_words: {stats['avg_chosen_words']}",
        f"- avg_rejected_words: {stats['avg_rejected_words']}",
        "",
        "## Label distribution",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label, count in stats["label_distribution"].items():
        lines.append(f"| {label} | {count} |")

    lines += ["", "## Source distribution", "", "| Source | Count |", "| --- | ---: |"]
    for source, count in stats["source_distribution"].items():
        lines.append(f"| {source} | {count} |")

    lines += ["", "## Rejection type distribution", "", "| Type | Count |", "| --- | ---: |"]
    for rejection_type, count in stats["rejection_type_distribution"].items():
        lines.append(f"| {rejection_type} | {count} |")

    lines.append("")
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
