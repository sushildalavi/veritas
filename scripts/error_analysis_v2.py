"""Failure analysis for the v2 oracle-vs-retrieved evaluation.

Builds on `scripts/eval_oracle_vs_retrieved_v2.py`: re-runs retrieval and the
per_passage_max verifier on the same dataset/config, then breaks the
aggregate metrics down by retrieval hit/miss, oracle-vs-retrieved agreement,
and dataset (FEVER vs SciFact), and dumps the worst failure cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import subprocess
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.config import load_project_settings
from data.schemas import ClaimEvidenceRecord, EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import iter_records, load_evidence_corpus, load_records, relevant_doc_ids
from models.model_router import ModelRouter
from retrieval.metrics import ndcg_at_k, recall_at_k
from serving.model_loader import _load_reranker_runtime, _load_retrieval_runtime

LABEL_ORDER = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
DEFAULT_SPLIT_PREFIXES = ("fever_test", "scifact_test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Failure analysis for oracle vs retrieved v2 eval.")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--suffix", default="_large")
    parser.add_argument("--split-prefixes", default="fever_test,scifact_test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--num-failure-examples", type=int, default=40)
    parser.add_argument("--profile-comparison-json", default="reports/retrieval_profile_comparison.json")
    parser.add_argument("--report-json", default="reports/error_analysis_650.json")
    parser.add_argument("--report-md", default="reports/error_analysis_650.md")
    parser.add_argument("--failures-jsonl", default="reports/failure_examples_650.jsonl")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings(args.config)
    checkpoint = args.checkpoint or settings.transformer_clean_checkpoint
    split_prefixes = _parse_prefixes(args.split_prefixes)

    records = _load_eval_records(
        Path(args.data_dir), suffix=args.suffix, split_prefixes=split_prefixes, max_examples=args.max_examples
    )
    corpus = load_evidence_corpus(Path(args.data_dir), suffix=args.suffix)
    retrieval_runtime = _load_retrieval_runtime(corpus, settings)
    reranker_runtime = _load_reranker_runtime(settings)

    retrieval_depth = max(args.top_k, settings.rrf_top_k, settings.rerank_top_k, settings.final_top_k, 50)

    retrieval_started = perf_counter()
    rankings: list[list[str]] = []
    retrieved_examples: list[list[EvidenceSpan]] = []
    for record in records:
        hits = retrieval_runtime.retriever.retrieve(record.claim, top_k=retrieval_depth)
        if reranker_runtime.reranker is not None:
            hits = list(reranker_runtime.reranker.rank(record.claim, hits))
        rankings.append([span.doc_id for span in hits])
        retrieved_examples.append(hits[: args.top_k])
    retrieval_runtime_seconds = round(perf_counter() - retrieval_started, 3)

    oracle_examples = [list(record.evidence) for record in records]
    labels = [_normalize_label(record.label) for record in records]
    claims = [record.claim for record in records]
    relevant_sets = [relevant_doc_ids(record) for record in records]
    datasets = [str(record.metadata.get("source", "unknown")) for record in records]

    prefer_deberta = settings.verifier_backend.lower() != "mock"
    per_passage_router = ModelRouter(
        verifier_checkpoint=checkpoint,
        prefer_deberta=prefer_deberta,
        aggregation_mode="per_passage_max",
        support_threshold=settings.support_threshold,
        refute_threshold=settings.refute_threshold,
    )

    oracle_predictions: list[str] = []
    oracle_logits: list[dict[str, float]] = []
    retrieved_predictions: list[str] = []
    retrieved_logits: list[dict[str, float]] = []
    for claim, oracle_evidence, retrieved_evidence in zip(claims, oracle_examples, retrieved_examples):
        oracle_result = per_passage_router.predict(claim, oracle_evidence)
        oracle_predictions.append(_normalize_label(oracle_result.verdict))
        oracle_logits.append(dict(oracle_result.logits))
        retrieved_result = per_passage_router.predict(claim, retrieved_evidence)
        retrieved_predictions.append(_normalize_label(retrieved_result.verdict))
        retrieved_logits.append(dict(retrieved_result.logits))

    top10_hit = [_has_overlap(ranking, relevant, 10) for ranking, relevant in zip(rankings, relevant_sets)]
    top50_hit = [_has_overlap(ranking, relevant, 50) for ranking, relevant in zip(rankings, relevant_sets)]

    confusion = confusion_matrix(labels, retrieved_predictions, labels=LABEL_ORDER).tolist()
    classification = classification_report(labels, retrieved_predictions, labels=LABEL_ORDER, output_dict=True, zero_division=0)
    per_class = {
        label: {
            "precision": round(float(classification[label]["precision"]), 4),
            "recall": round(float(classification[label]["recall"]), 4),
            "f1": round(float(classification[label]["f1-score"]), 4),
        }
        for label in LABEL_ORDER
    }
    overall = {
        "accuracy": round(float(accuracy_score(labels, retrieved_predictions)), 4),
        "macro_f1": round(float(f1_score(labels, retrieved_predictions, labels=LABEL_ORDER, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion,
        "confusion_matrix_labels": LABEL_ORDER,
        "per_class": per_class,
    }

    retrieval_hit_breakdown = {
        "top10": _subset_metrics(labels, retrieved_predictions, top10_hit),
        "top50": _subset_metrics(labels, retrieved_predictions, top50_hit),
    }

    oracle_vs_retrieved = _oracle_vs_retrieved_breakdown(labels, oracle_predictions, retrieved_predictions)

    per_dataset = _per_dataset_breakdown(
        labels=labels,
        oracle_predictions=oracle_predictions,
        retrieved_predictions=retrieved_predictions,
        datasets=datasets,
        rankings=rankings,
        relevant_sets=relevant_sets,
    )

    retrieval_profile_summary = _retrieval_profile_summary(Path(args.profile_comparison_json))

    failures = _build_failure_examples(
        records=records,
        labels=labels,
        oracle_predictions=oracle_predictions,
        retrieved_predictions=retrieved_predictions,
        retrieved_examples=retrieved_examples,
        retrieved_logits=retrieved_logits,
        oracle_logits=oracle_logits,
        top10_hit=top10_hit,
        top50_hit=top50_hit,
        datasets=datasets,
        max_examples=args.num_failure_examples,
    )

    report = {
        "headline": (
            f"error_analysis_650 complete: retrieved per_passage_max accuracy={overall['accuracy']:.3f}, "
            f"macro_f1={overall['macro_f1']:.3f}, retrieval_hit_top10_rate="
            f"{retrieval_hit_breakdown['top10']['hit_rate']:.3f}"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "checkpoint": checkpoint,
        "config_path": args.config,
        "dataset_source": _dataset_source(split_prefixes, args.suffix),
        "retrieval_backend": retrieval_runtime.retrieval_backend,
        "reranker_backend": reranker_runtime.reranker_backend,
        "top_k": args.top_k,
        "sample_size": len(records),
        "split_prefixes": list(split_prefixes),
        "suffix": args.suffix,
        "retrieval_runtime_seconds": retrieval_runtime_seconds,
        "label_distribution": _label_distribution(labels),
        "settings": {
            "verifier_aggregation": settings.verifier_aggregation,
            "support_threshold": settings.support_threshold,
            "refute_threshold": settings.refute_threshold,
        },
        "retrieved_per_passage_max": overall,
        "retrieval_hit_breakdown": retrieval_hit_breakdown,
        "oracle_vs_retrieved": oracle_vs_retrieved,
        "per_dataset": per_dataset,
        "retrieval_profile_summary": retrieval_profile_summary,
        "failure_examples_count": len(failures),
    }

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    _write_failures_jsonl(failures, Path(args.failures_jsonl))
    print(report["headline"])


def _has_overlap(ranking: list[str], relevant: list[str], k: int) -> bool:
    if not relevant:
        return False
    relevant_set = set(relevant)
    return any(doc_id in relevant_set for doc_id in ranking[:k])


def _subset_metrics(labels: list[str], predictions: list[str], mask: list[bool]) -> dict[str, object]:
    subset_labels = [label for label, keep in zip(labels, mask) if keep]
    subset_predictions = [pred for pred, keep in zip(predictions, mask) if keep]
    if not subset_labels:
        return {
            "count": 0,
            "hit_rate": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
        }
    return {
        "count": len(subset_labels),
        "hit_rate": round(len(subset_labels) / len(labels), 4),
        "accuracy": round(float(accuracy_score(subset_labels, subset_predictions)), 4),
        "macro_f1": round(float(f1_score(subset_labels, subset_predictions, labels=LABEL_ORDER, average="macro", zero_division=0)), 4),
    }


def _oracle_vs_retrieved_breakdown(labels: list[str], oracle_predictions: list[str], retrieved_predictions: list[str]) -> dict[str, object]:
    counts = {
        "oracle_correct_retrieved_correct": 0,
        "oracle_correct_retrieved_wrong": 0,
        "oracle_wrong_retrieved_wrong": 0,
        "oracle_wrong_retrieved_correct": 0,
    }
    for label, oracle_pred, retrieved_pred in zip(labels, oracle_predictions, retrieved_predictions):
        oracle_correct = oracle_pred == label
        retrieved_correct = retrieved_pred == label
        if oracle_correct and retrieved_correct:
            counts["oracle_correct_retrieved_correct"] += 1
        elif oracle_correct and not retrieved_correct:
            counts["oracle_correct_retrieved_wrong"] += 1
        elif not oracle_correct and not retrieved_correct:
            counts["oracle_wrong_retrieved_wrong"] += 1
        else:
            counts["oracle_wrong_retrieved_correct"] += 1
    total = len(labels)
    return {
        "counts": counts,
        "rates": {key: round(value / total, 4) for key, value in counts.items()} if total else counts,
        "total": total,
    }


def _per_dataset_breakdown(
    *,
    labels: list[str],
    oracle_predictions: list[str],
    retrieved_predictions: list[str],
    datasets: list[str],
    rankings: list[list[str]],
    relevant_sets: list[list[str]],
) -> dict[str, object]:
    breakdown: dict[str, object] = {}
    for dataset_name in sorted(set(datasets)):
        indices = [i for i, name in enumerate(datasets) if name == dataset_name]
        dataset_labels = [labels[i] for i in indices]
        dataset_oracle = [oracle_predictions[i] for i in indices]
        dataset_retrieved = [retrieved_predictions[i] for i in indices]
        dataset_rankings = [rankings[i] for i in indices]
        dataset_relevant = [relevant_sets[i] for i in indices]
        breakdown[dataset_name] = {
            "count": len(indices),
            "label_distribution": _label_distribution(dataset_labels),
            "retrieval_metrics": {
                "recall@10": round(mean(recall_at_k(r, rel, 10) for r, rel in zip(dataset_rankings, dataset_relevant)), 4),
                "recall@50": round(mean(recall_at_k(r, rel, 50) for r, rel in zip(dataset_rankings, dataset_relevant)), 4),
                "ndcg@10": round(mean(ndcg_at_k(r, rel, 10) for r, rel in zip(dataset_rankings, dataset_relevant)), 4),
            },
            "oracle_per_passage_max": {
                "accuracy": round(float(accuracy_score(dataset_labels, dataset_oracle)), 4),
                "macro_f1": round(float(f1_score(dataset_labels, dataset_oracle, labels=LABEL_ORDER, average="macro", zero_division=0)), 4),
            },
            "retrieved_per_passage_max": {
                "accuracy": round(float(accuracy_score(dataset_labels, dataset_retrieved)), 4),
                "macro_f1": round(float(f1_score(dataset_labels, dataset_retrieved, labels=LABEL_ORDER, average="macro", zero_division=0)), 4),
            },
        }
    return breakdown


def _retrieval_profile_summary(profile_comparison_path: Path) -> dict[str, object]:
    if not profile_comparison_path.exists():
        return {"status": "missing", "path": str(profile_comparison_path)}
    data = json.loads(profile_comparison_path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", {})
    measured = {
        name: profile
        for name, profile in profiles.items()
        if profile.get("status") == "measured"
    }
    if not measured:
        return {"status": "no_measured_profiles", "path": str(profile_comparison_path)}

    def _recall10(profile: dict[str, object]) -> float:
        return float(profile.get("retrieval_metrics", {}).get("recall@10", 0.0))

    def _macro_f1(profile: dict[str, object]) -> float:
        return float(profile.get("retrieved_per_passage_macro_f1", 0.0))

    best_recall = max(measured.items(), key=lambda item: _recall10(item[1]))
    best_macro_f1 = max(measured.items(), key=lambda item: _macro_f1(item[1]))
    baseline = measured.get("bm25_only")
    comparisons = {}
    if baseline is not None:
        for name in ("hybrid_bm25_dense", "hybrid_with_query_expansion", "hybrid_with_reranker", "hybrid_bm25_sentence_transformer"):
            profile = measured.get(name)
            if profile is None:
                continue
            comparisons[name] = {
                "recall@10_delta": round(_recall10(profile) - _recall10(baseline), 4),
                "per_passage_macro_f1_delta": round(_macro_f1(profile) - _macro_f1(baseline), 4),
            }
    return {
        "status": "ok",
        "source": str(profile_comparison_path),
        "max_examples": data.get("max_examples"),
        "best_recall@10": {"profile": best_recall[0], "recall@10": _recall10(best_recall[1])},
        "best_per_passage_macro_f1": {"profile": best_macro_f1[0], "per_passage_macro_f1": _macro_f1(best_macro_f1[1])},
        "bm25_only_baseline": {"recall@10": _recall10(baseline), "per_passage_macro_f1": _macro_f1(baseline)} if baseline else None,
        "vs_bm25_only": comparisons,
    }


def _build_failure_examples(
    *,
    records: list[ClaimEvidenceRecord],
    labels: list[str],
    oracle_predictions: list[str],
    retrieved_predictions: list[str],
    retrieved_examples: list[list[EvidenceSpan]],
    retrieved_logits: list[dict[str, float]],
    oracle_logits: list[dict[str, float]],
    top10_hit: list[bool],
    top50_hit: list[bool],
    datasets: list[str],
    max_examples: int,
) -> list[dict[str, object]]:
    failure_indices = [i for i, (label, pred) in enumerate(zip(labels, retrieved_predictions)) if label != pred]

    buckets: dict[tuple[str, str], list[int]] = {}
    for i in failure_indices:
        key = (datasets[i], labels[i])
        buckets.setdefault(key, []).append(i)

    ordered: list[int] = []
    bucket_iters = {key: iter(indices) for key, indices in buckets.items()}
    while len(ordered) < min(max_examples, len(failure_indices)):
        progressed = False
        for key in list(bucket_iters.keys()):
            try:
                ordered.append(next(bucket_iters[key]))
                progressed = True
                if len(ordered) >= min(max_examples, len(failure_indices)):
                    break
            except StopIteration:
                del bucket_iters[key]
        if not progressed:
            break

    failures = []
    for i in ordered:
        record = records[i]
        failures.append(
            {
                "claim_id": record.claim_id,
                "claim": record.claim,
                "dataset": datasets[i],
                "gold_label": labels[i],
                "predicted_label": retrieved_predictions[i],
                "oracle_prediction": oracle_predictions[i],
                "retrieved_prediction": retrieved_predictions[i],
                "gold_evidence_retrieved_top10": top10_hit[i],
                "gold_evidence_retrieved_top50": top50_hit[i],
                "retrieved_evidence": [
                    {"doc_id": span.doc_id, "title": span.title, "text": span.text}
                    for span in retrieved_examples[i]
                ],
                "retrieved_class_scores": {k: round(v, 4) for k, v in retrieved_logits[i].items()},
                "oracle_class_scores": {k: round(v, 4) for k, v in oracle_logits[i].items()},
            }
        )
    return failures


def _write_failures_jsonl(failures: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure) + "\n")


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


def _to_markdown(report: dict[str, object]) -> str:
    overall = report["retrieved_per_passage_max"]
    lines = [
        "# Error Analysis: Oracle vs Retrieved V2 (650-example full set)",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Git commit: `{report.get('git_commit')}`",
        f"- Config path: `{report.get('config_path')}`",
        f"- Dataset source: `{report.get('dataset_source')}`",
        f"- Retrieval backend: `{report['retrieval_backend']}`",
        f"- Reranker backend: `{report['reranker_backend']}`",
        f"- Sample size: {report['sample_size']}",
        f"- support_threshold: {report['settings']['support_threshold']}, refute_threshold: {report['settings']['refute_threshold']}",
        f"- Label distribution: {report['label_distribution']}",
        "",
        "## 1-2. Retrieved per_passage_max: confusion matrix and per-class metrics",
        "",
        f"- Accuracy: {overall['accuracy']}",
        f"- Macro F1: {overall['macro_f1']}",
        "",
        f"Confusion matrix (rows=gold, cols=predicted, order={overall['confusion_matrix_labels']}):",
        "",
    ]
    for row, label in zip(overall["confusion_matrix"], overall["confusion_matrix_labels"]):
        lines.append(f"- {label}: {row}")
    lines.extend(["", "| label | precision | recall | f1 |", "| --- | --- | --- | --- |"])
    for label in LABEL_ORDER:
        metrics = overall["per_class"][label]
        lines.append(f"| {label} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} |")

    lines.extend(["", "## 3. Retrieval-hit vs retrieval-miss breakdown", ""])
    lines.append("| gold evidence in | count | hit_rate | accuracy | macro_f1 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, metrics in report["retrieval_hit_breakdown"].items():
        lines.append(f"| top-{name[3:]} | {metrics['count']} | {metrics['hit_rate']} | {metrics['accuracy']} | {metrics['macro_f1']} |")
    miss10 = report["sample_size"] - report["retrieval_hit_breakdown"]["top10"]["count"]
    miss50 = report["sample_size"] - report["retrieval_hit_breakdown"]["top50"]["count"]
    lines.append(f"\n_Misses: {miss10} examples missing gold evidence in top-10, {miss50} missing in top-50._")

    lines.extend(["", "## 4. Oracle vs retrieved agreement (per_passage_max)", ""])
    ovr = report["oracle_vs_retrieved"]
    lines.append("| outcome | count | rate |")
    lines.append("| --- | --- | --- |")
    for key, count in ovr["counts"].items():
        lines.append(f"| {key} | {count} | {ovr['rates'][key]} |")

    lines.extend(["", "## 5. Per-dataset breakdown", ""])
    lines.append("| dataset | count | label_distribution | recall@10 | recall@50 | ndcg@10 | oracle acc/F1 | retrieved acc/F1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for dataset_name, metrics in report["per_dataset"].items():
        rm = metrics["retrieval_metrics"]
        om = metrics["oracle_per_passage_max"]
        rt = metrics["retrieved_per_passage_max"]
        lines.append(
            f"| {dataset_name} | {metrics['count']} | {metrics['label_distribution']} | {rm['recall@10']} | {rm['recall@50']} | "
            f"{rm['ndcg@10']} | {om['accuracy']}/{om['macro_f1']} | {rt['accuracy']}/{rt['macro_f1']} |"
        )

    lines.extend(["", "## 6. Retrieval profile comparison (50-example slice)", ""])
    profile_summary = report["retrieval_profile_summary"]
    if profile_summary.get("status") == "ok":
        lines.append(f"- Source: `{profile_summary['source']}` (max_examples={profile_summary['max_examples']})")
        lines.append(f"- Best recall@10: `{profile_summary['best_recall@10']['profile']}` ({profile_summary['best_recall@10']['recall@10']})")
        lines.append(
            f"- Best retrieved per-passage macro-F1: `{profile_summary['best_per_passage_macro_f1']['profile']}` "
            f"({profile_summary['best_per_passage_macro_f1']['per_passage_macro_f1']})"
        )
        baseline = profile_summary["bm25_only_baseline"]
        lines.append(f"- bm25_only baseline: recall@10={baseline['recall@10']}, per_passage_macro_f1={baseline['per_passage_macro_f1']}")
        lines.append("")
        lines.append("| profile vs bm25_only | recall@10 delta | per_passage_macro_f1 delta |")
        lines.append("| --- | --- | --- |")
        for name, delta in profile_summary["vs_bm25_only"].items():
            lines.append(f"| {name} | {delta['recall@10_delta']} | {delta['per_passage_macro_f1_delta']} |")
    else:
        lines.append(f"- {profile_summary.get('status')}")

    lines.extend(
        [
            "",
            "## 7-8. Failure examples",
            "",
            f"- {report['failure_examples_count']} failure examples saved to `reports/failure_examples_650.jsonl` "
            "(claim, gold/predicted/oracle/retrieved labels, retrieved evidence texts, retrieval hit flags, class scores).",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
