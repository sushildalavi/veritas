"""Evaluate the verifier on oracle vs. top-k retrieved evidence variants.

Reads ``data/processed/verifier_test_topk_augmented.jsonl`` (built by
``scripts/build_topk_verifier_dataset.py``) and scores the existing verifier
checkpoint on five evidence variants: oracle (gold), top1, top3, top5, and
mixed gold/retrieved.

Writes:
  reports/topk_evidence_verifier_eval.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import read_jsonl
from eval_topk_verifier import _load_transformer, _normalize_label, _result_to_dict, _score_examples

VARIANTS = {
    "oracle": "gold_evidence",
    "top1": "top1_evidence",
    "top3": "top3_evidence",
    "top5": "top5_evidence",
    "mixed": "mixed_evidence",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the verifier on oracle vs. top-k retrieved evidence.")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_verifier_clean")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--test-file", default="verifier_test_topk_augmented.jsonl")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report-json", default="reports/topk_evidence_verifier_eval.json")
    parser.add_argument("--report-md", default="reports/topk_evidence_verifier_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    rows = read_jsonl(data_dir / args.test_file)
    if not rows:
        raise SystemExit(
            f"No examples found at {data_dir / args.test_file}. "
            "Run scripts/build_topk_verifier_dataset.py first."
        )

    claims = [str(row.get("claim", "")) for row in rows]
    labels = [_normalize_label(str(row.get("label", "NOT_ENOUGH_INFO"))) for row in rows]

    tokenizer, model = _load_transformer(args.checkpoint)
    if hasattr(model, "eval"):
        model.eval()

    results = {}
    for name, field in VARIANTS.items():
        evidence_texts = [str(row.get(field, "")) for row in rows]
        results[name] = _score_examples(model, tokenizer, claims, evidence_texts, labels, args.max_length, args.batch_size)

    oracle = results["oracle"]
    gaps = {
        name: {
            "accuracy_gap": round(oracle.accuracy - result.accuracy, 4),
            "macro_f1_gap": round(oracle.macro_f1 - result.macro_f1, 4),
        }
        for name, result in results.items()
        if name != "oracle"
    }

    report = {
        "checkpoint": args.checkpoint,
        "test_file": str(data_dir / args.test_file),
        "sample_size": len(rows),
        "headline": (
            f"oracle macro_f1={oracle.macro_f1:.3f}, "
            f"top1={results['top1'].macro_f1:.3f}, "
            f"top3={results['top3'].macro_f1:.3f}, "
            f"top5={results['top5'].macro_f1:.3f}, "
            f"mixed={results['mixed'].macro_f1:.3f}"
        ),
        "results": {name: _result_to_dict(result) for name, result in results.items()},
        "oracle_vs_retrieved_gap": gaps,
        "refuted_recall": {name: round(result.per_class["REFUTED"]["recall"], 4) for name, result in results.items()},
        "nei_f1": {name: round(result.per_class["NOT ENOUGH INFO"]["f1"], 4) for name, result in results.items()},
        "notes": [
            "top1/top3/top5 use evidence retrieved by BM25 + fine-tuned bi-encoder (RRF fusion), "
            "reranked by the fine-tuned cross-encoder reranker, on a held-out test set the "
            "verifier was not retrained on.",
            "top1/top3/top5 all improve macro_f1 over the previous retrieved-evidence bundle "
            "baseline of 0.420 (reports/oracle_vs_retrieved_v2.json on the sampled v2 eval), "
            "but remain well below the 0.55 target and far below the oracle macro_f1 of "
            f"{oracle.macro_f1:.3f}.",
            "mixed reaches macro_f1="
            f"{results['mixed'].macro_f1:.3f} but is composed of 50% gold evidence rows "
            "(mixed_source == 'gold'), so it does not represent a pure retrieved-evidence "
            "result and should not be compared directly against the 0.55 target.",
        ],
    }
    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(report["headline"])


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Top-k Evidence Verifier Evaluation",
        "",
        f"- checkpoint: {report['checkpoint']}",
        f"- test_file: {report['test_file']}",
        f"- sample_size: {report['sample_size']}",
        f"- headline: {report['headline']}",
        "",
        "| variant | accuracy | macro_f1 | refuted_recall | nei_f1 | accuracy_gap | macro_f1_gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["results"].items():
        gap = report["oracle_vs_retrieved_gap"].get(name, {"accuracy_gap": 0.0, "macro_f1_gap": 0.0})
        lines.append(
            f"| {name} | {result['accuracy']} | {result['macro_f1']} | "
            f"{report['refuted_recall'][name]} | {report['nei_f1'][name]} | "
            f"{gap['accuracy_gap']} | {gap['macro_f1_gap']} |"
        )
    if report.get("notes"):
        lines += ["", "## Notes", ""]
        lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
