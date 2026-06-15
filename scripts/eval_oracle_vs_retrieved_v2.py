"""Evaluate retrieval ceilings and oracle-vs-retrieved verifier behavior."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.config import ProjectSettings, load_project_settings
from data.schemas import ClaimEvidenceRecord, EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import iter_records, load_evidence_corpus, load_records, relevant_doc_ids
from models.model_router import ModelRouter
from retrieval.metrics import ndcg_at_k, recall_at_k
from serving.model_loader import _load_retrieval_runtime, _load_reranker_runtime

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
DEFAULT_SPLIT_PREFIXES = ("fever_test", "scifact_test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate oracle vs retrieved verifier performance.")
    parser.add_argument("--config", default="configs/serving_advanced.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--suffix", default="_large")
    parser.add_argument("--split-prefixes", default="fever_test,scifact_test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--threshold-support", type=float, default=None)
    parser.add_argument("--threshold-refute", type=float, default=None)
    parser.add_argument("--gate-threshold", type=float, default=None, help="Relevance gate lexical overlap threshold (None = disabled)")
    parser.add_argument("--report-json", default="reports/oracle_vs_retrieved_v2.json")
    parser.add_argument("--report-md", default="reports/oracle_vs_retrieved_v2.md")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings(args.config)
    checkpoint = args.checkpoint or settings.transformer_clean_checkpoint
    split_prefixes = _parse_prefixes(args.split_prefixes)
    report = evaluate_oracle_vs_retrieved_v2(
        settings=settings,
        checkpoint=checkpoint,
        data_dir=Path(args.data_dir),
        suffix=args.suffix,
        split_prefixes=split_prefixes,
        top_k=args.top_k,
        max_examples=args.max_examples,
        config_path=args.config,
        support_threshold=args.threshold_support,
        refute_threshold=args.threshold_refute,
        gate_threshold=args.gate_threshold,
    )
    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(report["headline"])


def evaluate_oracle_vs_retrieved_v2(
    *,
    settings: ProjectSettings,
    checkpoint: str,
    data_dir: Path,
    suffix: str,
    split_prefixes: tuple[str, ...] = DEFAULT_SPLIT_PREFIXES,
    top_k: int,
    max_examples: int = 0,
    config_path: str | None = None,
    support_threshold: float | None = None,
    refute_threshold: float | None = None,
    gate_threshold: float | None = None,
) -> dict[str, object]:
    support_threshold = settings.support_threshold if support_threshold is None else support_threshold
    refute_threshold = settings.refute_threshold if refute_threshold is None else refute_threshold
    gate_threshold = settings.relevance_gate_threshold if gate_threshold is None else gate_threshold
    records = _load_eval_records(data_dir, suffix=suffix, split_prefixes=split_prefixes, max_examples=max_examples)
    corpus = load_evidence_corpus(data_dir, suffix=suffix)
    retrieval_runtime = _load_retrieval_runtime(corpus, settings)
    reranker_runtime = _load_reranker_runtime(settings)

    retrieval_rankings: list[list[str]] = []
    retrieved_examples: list[list[EvidenceSpan]] = []
    oracle_examples = [list(record.evidence) for record in records]
    labels = [_normalize_label(record.label) for record in records]
    claims = [record.claim for record in records]

    retrieval_started = perf_counter()
    retrieval_depth = max(top_k, settings.rrf_top_k, settings.rerank_top_k, settings.final_top_k, 50)
    for record in records:
        hits = retrieval_runtime.retriever.retrieve(record.claim, top_k=retrieval_depth)
        if reranker_runtime.reranker is not None:
            hits = list(reranker_runtime.reranker.rank(record.claim, hits))
        retrieval_rankings.append([span.doc_id for span in hits])
        retrieved_examples.append(hits[:top_k])
    retrieval_latency = perf_counter() - retrieval_started

    prefer_deberta = settings.verifier_backend.lower() != "mock"
    bundle_router = ModelRouter(
        verifier_checkpoint=checkpoint,
        prefer_deberta=prefer_deberta,
        aggregation_mode="bundle",
        support_threshold=support_threshold,
        refute_threshold=refute_threshold,
        relevance_gate_threshold=gate_threshold,
    )
    per_passage_router = ModelRouter(
        verifier_checkpoint=checkpoint,
        prefer_deberta=prefer_deberta,
        aggregation_mode="per_passage_max",
        support_threshold=support_threshold,
        refute_threshold=refute_threshold,
        relevance_gate_threshold=gate_threshold,
    )

    oracle_bundle = _evaluate_router(bundle_router, claims, labels, oracle_examples)
    retrieved_bundle = _evaluate_router(bundle_router, claims, labels, retrieved_examples)
    oracle_per_passage = _evaluate_router(per_passage_router, claims, labels, oracle_examples)
    retrieved_per_passage = _evaluate_router(per_passage_router, claims, labels, retrieved_examples)

    relevant_sets = [relevant_doc_ids(record) for record in records]
    retrieval_metrics = _retrieval_metrics(retrieval_rankings, relevant_sets)
    report = {
        "headline": (
            f"oracle_vs_retrieved_v2 complete: retrieval recall@10={retrieval_metrics['recall@10']:.3f}, "
            f"retrieved macro_f1 bundle={retrieved_bundle['macro_f1']:.3f}, "
            f"per_passage={retrieved_per_passage['macro_f1']:.3f}"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "checkpoint": checkpoint,
        "config_path": config_path,
        "dataset_source": _dataset_source(split_prefixes, suffix),
        "retrieval_backend": retrieval_runtime.retrieval_backend,
        "reranker_backend": reranker_runtime.reranker_backend,
        "top_k": top_k,
        "sample_size": len(records),
        "split_prefixes": list(split_prefixes),
        "suffix": suffix,
        "retrieval_runtime_seconds": round(retrieval_latency, 3),
        "retrieval_metrics": retrieval_metrics,
        "label_distribution": _label_distribution(labels),
        "settings": {
            "bm25_top_k": settings.bm25_top_k,
            "dense_top_k": settings.dense_top_k,
            "title_top_k": settings.title_top_k,
            "query_expansion_top_k": settings.query_expansion_top_k,
            "rrf_top_k": settings.rrf_top_k,
            "rerank_top_k": settings.rerank_top_k,
            "final_top_k": settings.final_top_k,
            "verifier_aggregation": settings.verifier_aggregation,
            "support_threshold": support_threshold,
            "refute_threshold": refute_threshold,
        },
        "oracle": {
            "bundle": oracle_bundle,
            "per_passage_max": oracle_per_passage,
        },
        "retrieved": {
            "bundle": retrieved_bundle,
            "per_passage_max": retrieved_per_passage,
        },
        "delta_from_oracle": {
            "bundle": _delta(oracle_bundle, retrieved_bundle),
            "per_passage_max": _delta(oracle_per_passage, retrieved_per_passage),
        },
    }
    return report


def _load_eval_records(
    data_dir: Path,
    *,
    suffix: str,
    split_prefixes: tuple[str, ...],
    max_examples: int,
) -> list[ClaimEvidenceRecord]:
    split_names = [f"{prefix}{suffix}" for prefix in split_prefixes]
    records_by_split = load_records(data_dir, split_names=split_names)
    records = iter_records(records_by_split, split_prefixes)
    filtered = [record for record in records if record.claim.strip()]
    if max_examples > 0:
        return filtered[:max_examples]
    return filtered


def _evaluate_router(
    router: ModelRouter,
    claims: list[str],
    labels: list[str],
    evidence_sets: list[list[EvidenceSpan]],
) -> dict[str, object]:
    predictions: list[str] = []
    confidences: list[float] = []
    latencies_ms: list[float] = []
    for claim, evidence in zip(claims, evidence_sets):
        started = perf_counter()
        result = router.predict(claim, evidence)
        latencies_ms.append((perf_counter() - started) * 1000.0)
        predictions.append(_normalize_label(result.verdict))
        confidences.append(result.confidence)
    report = classification_report(labels, predictions, labels=LABEL_ORDER, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=LABEL_ORDER).tolist()
    nei_total = sum(1 for label in labels if label == "NOT ENOUGH INFO")
    nei_false_positives = sum(1 for truth, pred in zip(labels, predictions) if truth == "NOT ENOUGH INFO" and pred != "NOT ENOUGH INFO")
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, labels=LABEL_ORDER, average="macro", zero_division=0)), 4),
        "average_confidence": round(mean(confidences), 4) if confidences else 0.0,
        "latency_ms": {
            "p50": round(_percentile(latencies_ms, 50), 4),
            "p95": round(_percentile(latencies_ms, 95), 4),
        },
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
            }
            for label in LABEL_ORDER
        },
        "confusion_matrix": matrix,
        "nei_false_positive_rate": round(nei_false_positives / nei_total, 4) if nei_total else 0.0,
    }


def _retrieval_metrics(
    retrieval_rankings: list[list[str]],
    relevant_sets: list[list[str]],
) -> dict[str, float]:
    metrics = {
        f"recall@{k}": round(mean(recall_at_k(doc_ids, relevant, k) for doc_ids, relevant in zip(retrieval_rankings, relevant_sets)), 4)
        for k in (1, 3, 5, 10, 50)
    }
    metrics["ndcg@5"] = round(mean(ndcg_at_k(doc_ids, relevant, 5) for doc_ids, relevant in zip(retrieval_rankings, relevant_sets)), 4)
    metrics["ndcg@10"] = round(mean(ndcg_at_k(doc_ids, relevant, 10) for doc_ids, relevant in zip(retrieval_rankings, relevant_sets)), 4)
    return metrics


def _label_distribution(labels: list[str]) -> dict[str, int]:
    distribution = {label: 0 for label in LABEL_ORDER}
    for label in labels:
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ")
    if normalized in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if normalized in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def _parse_prefixes(raw: str) -> tuple[str, ...]:
    prefixes = tuple(part.strip() for part in raw.split(",") if part.strip())
    return prefixes or DEFAULT_SPLIT_PREFIXES


def _dataset_source(split_prefixes: tuple[str, ...], suffix: str) -> str:
    joined = ", ".join(f"{prefix}{suffix}" for prefix in split_prefixes)
    return f"structured records from {joined}"


def _git_commit_hash() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            or None
        )
    except Exception:
        return None


def _delta(oracle: dict[str, object], retrieved: dict[str, object]) -> dict[str, float]:
    return {
        "accuracy": round(float(oracle["accuracy"]) - float(retrieved["accuracy"]), 4),
        "macro_f1": round(float(oracle["macro_f1"]) - float(retrieved["macro_f1"]), 4),
    }


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percentile / 100.0) * (len(ordered) - 1))
    return float(ordered[index])


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Oracle vs Retrieved V2",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Git commit: `{report.get('git_commit')}`",
        f"- Config path: `{report.get('config_path')}`",
        f"- Dataset source: `{report.get('dataset_source')}`",
        f"- Retrieval backend: `{report['retrieval_backend']}`",
        f"- Reranker backend: `{report['reranker_backend']}`",
        f"- Split prefixes: `{', '.join(report['split_prefixes'])}`",
        f"- Suffix: `{report['suffix']}`",
        f"- Sample size: {report['sample_size']}",
        f"- Retrieval runtime (s): {report['retrieval_runtime_seconds']}",
        "",
        "## Retrieval metrics",
        "",
    ]
    for key, value in report["retrieval_metrics"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Verifier metrics", ""])
    lines.append("| mode | evidence | accuracy | macro_f1 | nei_false_positive_rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for evidence_name in ("oracle", "retrieved"):
        for mode_name, metrics in report[evidence_name].items():
            lines.append(
                f"| {mode_name} | {evidence_name} | {metrics['accuracy']} | {metrics['macro_f1']} | {metrics['nei_false_positive_rate']} |"
            )
    lines.extend(["", "## Delta from oracle", ""])
    for mode_name, metrics in report["delta_from_oracle"].items():
        lines.append(f"- {mode_name}: accuracy gap={metrics['accuracy']}, macro_f1 gap={metrics['macro_f1']}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
