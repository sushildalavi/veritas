"""Evaluate the verifier with oracle vs. top-k retrieved evidence.

This report is meant to measure whether feeding more retrieved evidence closes
the oracle-to-retrieved gap compared with the existing top-1 report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evidence_formatting import format_verifier_text, render_evidence
from core.config import ProjectSettings
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_evidence_corpus, read_jsonl
from retrieval.bm25 import BM25Retriever
from serving.model_loader import load_pipeline

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}


@dataclass(frozen=True)
class EvaluationResult:
    sample_size: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    latency_ms_per_example: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the verifier with oracle and top-k retrieved evidence.")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_verifier_clean")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--test-file", default="verifier_test.jsonl")
    parser.add_argument("--evidence-corpus-suffix", default="_large")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-ks", default="1,3,5")
    parser.add_argument("--retrieval-mode", choices=["bm25", "research"], default="bm25")
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--report-json", default="reports/topk_verifier_eval.json")
    parser.add_argument("--report-md", default="reports/topk_verifier_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    rows = read_jsonl(data_dir / args.test_file)
    if not rows:
        raise SystemExit(f"No test examples found at {data_dir / args.test_file}")

    top_ks = _parse_top_ks(args.top_ks)
    report = evaluate_topk_verifier(
        checkpoint=args.checkpoint,
        data_dir=data_dir,
        test_file=args.test_file,
        evidence_corpus_suffix=args.evidence_corpus_suffix,
        top_ks=top_ks,
        retrieval_mode=args.retrieval_mode,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_errors=args.max_errors,
    )
    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(report["headline"])


def evaluate_topk_verifier(
    *,
    checkpoint: str,
    data_dir: Path,
    test_file: str,
    evidence_corpus_suffix: str,
    top_ks: list[int],
    retrieval_mode: str,
    max_length: int,
    batch_size: int,
    max_errors: int,
) -> dict[str, object]:
    rows = read_jsonl(data_dir / test_file)
    claims = [str(row.get("claim", "")) for row in rows]
    labels = [_normalize_label(str(row.get("label", "NOT_ENOUGH_INFO"))) for row in rows]
    oracle_evidence = [str(row.get("evidence", "")) for row in rows]

    started = perf_counter()
    corpus = load_evidence_corpus(data_dir, suffix=evidence_corpus_suffix)
    retriever, ranker, retrieval_backend = _build_retrieval_stack(corpus, retrieval_mode)
    retrieved_by_k: dict[int, list[str]] = {k: [] for k in top_ks}
    for claim in claims:
        hits = retriever.retrieve(claim, top_k=max(top_ks))
        if ranker is not None:
            try:
                hits = list(ranker.rank(claim, hits))
            except Exception:
                pass
        for k in top_ks:
            retrieved_by_k[k].append(_join_evidence(hits[:k]))
    retrieval_runtime = round(perf_counter() - started, 3)

    tokenizer, model = _load_transformer(checkpoint)
    if hasattr(model, "eval"):
        model.eval()

    oracle_metrics = _score_examples(model, tokenizer, claims, oracle_evidence, labels, max_length, batch_size)
    topk_metrics: dict[int, EvaluationResult] = {}
    for k in top_ks:
        topk_metrics[k] = _score_examples(model, tokenizer, claims, retrieved_by_k[k], labels, max_length, batch_size)

    gaps = {
        f"top_{k}": {
            "accuracy_gap": round(oracle_metrics.accuracy - result.accuracy, 4),
            "macro_f1_gap": round(oracle_metrics.macro_f1 - result.macro_f1, 4),
        }
        for k, result in topk_metrics.items()
    }
    best_k = max(topk_metrics, key=lambda k: topk_metrics[k].macro_f1)

    report = {
        "headline": (
            f"Top-k verifier evaluation complete. oracle macro_f1={oracle_metrics.macro_f1:.3f}, "
            f"best top_k={best_k} macro_f1={topk_metrics[best_k].macro_f1:.3f}"
        ),
        "checkpoint": checkpoint,
        "evidence_corpus_source": f"data/processed/evidence_corpus{evidence_corpus_suffix}.jsonl",
        "retrieval_mode": retrieval_mode,
        "retrieval_backend": retrieval_backend,
        "sample_size": len(rows),
        "retrieval_runtime_seconds": retrieval_runtime,
        "oracle": _result_to_dict(oracle_metrics),
        "top_k": {f"top_{k}": _result_to_dict(result) for k, result in topk_metrics.items()},
        "oracle_vs_retrieved_gap": gaps,
        "best_top_k": best_k,
        "top_errors": _top_errors(rows, claims, labels, retrieved_by_k[best_k], _predict_labels(model, tokenizer, claims, retrieved_by_k[best_k], max_length, batch_size), max_errors),
    }
    return report


def _build_retrieval_stack(corpus, retrieval_mode: str):  # noqa: ANN001
    if retrieval_mode == "research":
        settings = ProjectSettings(
            verifier_backend="auto",
            verifier_checkpoint="checkpoints/transformer_verifier_clean",
            retrieval_backend="bm25_sentence_transformer_hybrid",
            use_neural_retrieval=True,
            reranker_backend="cross_encoder",
            use_cross_encoder=True,
            explanation_mode="template",
        )
        pipeline = load_pipeline(settings=settings)
        return pipeline.retriever, pipeline.reranker, pipeline.retrieval_backend
    return BM25Retriever(corpus), None, "bm25_only"


def _score_examples(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    claims: list[str],
    evidence_texts: list[str],
    labels: list[str],
    max_length: int,
    batch_size: int,
) -> EvaluationResult:
    started = perf_counter()
    preds = _predict_labels(model, tokenizer, claims, evidence_texts, max_length, batch_size)
    latency_seconds = (perf_counter() - started) / max(len(labels), 1)
    y_true = [LABEL_TO_ID[label] for label in labels]
    report = classification_report(
        y_true,
        preds,
        target_names=LABEL_ORDER,
        labels=list(range(len(LABEL_ORDER))),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, preds, labels=list(range(len(LABEL_ORDER))))
    return EvaluationResult(
        sample_size=len(labels),
        accuracy=float(accuracy_score(y_true, preds)),
        macro_f1=float(f1_score(y_true, preds, average="macro")),
        per_class={
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
            }
            for label in LABEL_ORDER
        },
        confusion_matrix=matrix.tolist(),
        latency_ms_per_example=round(latency_seconds * 1000.0, 4),
    )


def _predict_labels(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    claims: list[str],
    evidence_texts: list[str],
    max_length: int,
    batch_size: int,
) -> list[int]:
    texts = [_build_input(claim, evidence) for claim, evidence in zip(claims, evidence_texts)]
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            logits = model(**encoded).logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
    return predictions


def _build_input(claim: str, evidence: str) -> str:
    return format_verifier_text(claim, evidence)


def _load_transformer(checkpoint: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    return tokenizer, model


def _join_evidence(passages: list[object]) -> str:
    typed_passages = [span for span in passages if hasattr(span, "text")]
    return render_evidence(typed_passages, style="plain", include_title=False, canonicalize=True)


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def _top_errors(
    rows: list[dict],
    claims: list[str],
    labels: list[str],
    evidence_texts: list[str],
    preds: list[int],
    limit: int,
) -> list[dict[str, object]]:
    errors = []
    for row, claim, true_label, ev, pred in zip(rows, claims, labels, evidence_texts, preds):
        pred_label = LABEL_ORDER[pred]
        if pred_label == true_label:
            continue
        errors.append(
            {
                "claim_id": row.get("claim_id", ""),
                "claim": claim,
                "true_label": true_label,
                "predicted_label": pred_label,
                "evidence": ev[:300],
            }
        )
        if len(errors) >= limit:
            break
    return errors


def _parse_top_ks(value: str) -> list[int]:
    ks = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        ks.append(int(piece))
    return sorted(set(ks))


def _result_to_dict(result: EvaluationResult) -> dict[str, object]:
    return {
        "sample_size": result.sample_size,
        "accuracy": round(result.accuracy, 4),
        "macro_f1": round(result.macro_f1, 4),
        "per_class": result.per_class,
        "confusion_matrix": result.confusion_matrix,
        "latency_ms_per_example": round(result.latency_ms_per_example, 4),
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Top-k Verifier Evaluation",
        "",
        f"- checkpoint: {report['checkpoint']}",
        f"- evidence_corpus_source: {report['evidence_corpus_source']}",
        f"- sample_size: {report['sample_size']}",
        f"- retrieval_runtime_seconds: {report['retrieval_runtime_seconds']}",
        f"- best_top_k: {report['best_top_k']}",
        "",
        "| setting | accuracy | macro_f1 | latency_ms_per_example |",
        "| --- | ---: | ---: | ---: |",
    ]
    oracle = report["oracle"]
    lines.append(
        f"| oracle | {oracle['accuracy']} | {oracle['macro_f1']} | {oracle['latency_ms_per_example']} |"
    )
    for name, result in report["top_k"].items():
        lines.append(
            f"| {name} | {result['accuracy']} | {result['macro_f1']} | {result['latency_ms_per_example']} |"
        )
    lines += ["", "## Oracle vs. retrieved gap", ""]
    for name, gap in report["oracle_vs_retrieved_gap"].items():
        lines.append(f"- {name}: accuracy_gap={gap['accuracy_gap']} macro_f1_gap={gap['macro_f1_gap']}")
    if report.get("top_errors"):
        lines += ["", "## Top errors", ""]
        for error in report["top_errors"]:
            lines.append(
                f"- claim_id={error['claim_id']} true={error['true_label']} pred={error['predicted_label']}: "
                f"\"{error['claim']}\" | evidence: \"{error['evidence']}\""
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
