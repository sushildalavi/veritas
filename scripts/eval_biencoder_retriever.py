"""Evaluate the fine-tuned bi-encoder retriever against BM25 and generic dense baselines.

Compares four retrieval strategies on the same query set:

  - bm25: lexical BM25 baseline
  - dense_generic: off-the-shelf sentence-transformers/all-MiniLM-L6-v2
  - dense_finetuned: bi-encoder fine-tuned by scripts/train_biencoder_retriever.py
  - hybrid_finetuned: BM25 + fine-tuned bi-encoder, fused with reciprocal-rank fusion

Writes:
  reports/biencoder_retriever_eval.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_evidence_corpus, load_records, relevant_doc_ids
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever, load_embedder
from retrieval.hybrid import HybridRetriever
from retrieval.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned bi-encoder retriever.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--file-suffix", default="_large")
    parser.add_argument("--checkpoint", default="checkpoints/biencoder_retriever")
    parser.add_argument("--generic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-json", default="reports/biencoder_retriever_eval.json")
    parser.add_argument("--output-md", default="reports/biencoder_retriever_eval.md")
    return parser


def evaluate_biencoder_retriever(
    *,
    data_dir: Path,
    split: str,
    max_queries: int,
    file_suffix: str,
    checkpoint: str,
    generic_model: str,
) -> dict[str, object]:
    started = perf_counter()
    records_by_split = load_records(data_dir, suffix=file_suffix) if file_suffix else load_records(data_dir)
    corpus = load_evidence_corpus(data_dir, suffix=file_suffix) if file_suffix else load_evidence_corpus(data_dir)

    records = []
    for split_name, split_records in records_by_split.items():
        if split_name.endswith(f"_{split}"):
            records.extend(split_records)
    records = records[:max_queries]
    if not records:
        raise ValueError(f"No records found for split={split!r}")

    notes: list[str] = []
    bm25 = BM25Retriever(corpus)

    strategies: dict[str, tuple[object, str, str]] = {
        "bm25": (bm25, "bm25", "lexical"),
    }
    try:
        generic_embedder = load_embedder("sentence-transformers", generic_model, allow_fallback=False)
        strategies["dense_generic"] = (DenseRetriever(corpus, embedder=generic_embedder), generic_model, "semantic")
    except Exception as exc:
        notes.append(f"Generic dense retrieval unavailable: {type(exc).__name__}: {exc}")

    try:
        finetuned_embedder = load_embedder("sentence-transformers", checkpoint, allow_fallback=False)
        finetuned_dense = DenseRetriever(corpus, embedder=finetuned_embedder)
        strategies["dense_finetuned"] = (finetuned_dense, checkpoint, "semantic")
        strategies["hybrid_finetuned"] = (HybridRetriever(bm25, finetuned_dense), f"bm25+{checkpoint}", "hybrid")
    except Exception as exc:
        notes.append(f"Fine-tuned bi-encoder retrieval unavailable: {type(exc).__name__}: {exc}")

    relevant_sets = [relevant_doc_ids(record) for record in records]
    strategy_metrics: dict[str, dict[str, float | str]] = {}
    for name, (retriever, model_name, backend_kind) in strategies.items():
        strategy_start = perf_counter()
        rankings = [retriever.retrieve(record.claim, top_k=10) for record in records]
        doc_rankings = [[span.doc_id for span in ranking] for ranking in rankings]
        strategy_metrics[name] = {
            "recall@1": round(mean(recall_at_k(ids, relevant, 1) for ids, relevant in zip(doc_rankings, relevant_sets)), 4),
            "recall@5": round(mean(recall_at_k(ids, relevant, 5) for ids, relevant in zip(doc_rankings, relevant_sets)), 4),
            "recall@10": round(mean(recall_at_k(ids, relevant, 10) for ids, relevant in zip(doc_rankings, relevant_sets)), 4),
            "mrr": round(mean_reciprocal_rank(doc_rankings, relevant_sets), 4),
            "ndcg@10": round(mean(ndcg_at_k(ids, relevant, 10) for ids, relevant in zip(doc_rankings, relevant_sets)), 4),
            "latency_seconds": round((perf_counter() - strategy_start) / max(len(records), 1), 4),
            "backend": backend_kind,
            "model_name": model_name,
        }

    if "dense_generic" in strategy_metrics and "dense_finetuned" in strategy_metrics:
        generic = strategy_metrics["dense_generic"]
        finetuned = strategy_metrics["dense_finetuned"]
        for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"):
            delta = round(float(finetuned[metric]) - float(generic[metric]), 4)
            direction = "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged")
            notes.append(f"Fine-tuned vs generic dense {metric}: {direction} by {abs(delta)} ({generic[metric]} -> {finetuned[metric]}).")

    return {
        "split": split,
        "query_count": len(records),
        "corpus_size": len(corpus),
        "runtime_seconds": round(perf_counter() - started, 3),
        "strategies": strategy_metrics,
        "notes": notes,
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Bi-Encoder Retriever Evaluation",
        "",
        f"- split: {report['split']}",
        f"- query_count: {report['query_count']}",
        f"- corpus_size: {report['corpus_size']}",
        f"- runtime_seconds: {report['runtime_seconds']}",
        "",
        "| strategy | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | latency_seconds | backend | model_name |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for name, metrics in report["strategies"].items():
        lines.append(
            f"| {name} | {metrics['recall@1']} | {metrics['recall@5']} | {metrics['recall@10']} | "
            f"{metrics['mrr']} | {metrics['ndcg@10']} | {metrics['latency_seconds']} | {metrics['backend']} | {metrics['model_name']} |"
        )
    if report.get("notes"):
        lines += ["", "## Notes", ""]
        lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    report = evaluate_biencoder_retriever(
        data_dir=Path(args.data_dir),
        split=args.split,
        max_queries=args.max_queries,
        file_suffix=args.file_suffix,
        checkpoint=args.checkpoint,
        generic_model=args.generic_model,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    write_report(report, output_json)
    output_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote bi-encoder retriever eval to {output_json}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
