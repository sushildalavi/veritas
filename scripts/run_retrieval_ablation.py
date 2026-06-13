"""Research-grade retrieval ablation for the Mac-local Veritas stack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_evidence_corpus, load_records, relevant_doc_ids
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever, load_embedder
from retrieval.hybrid import HybridRetriever
from retrieval.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


@dataclass(frozen=True)
class StrategyResult:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    latency_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval ablations across sparse, dense, and hybrid backends.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--file-suffix", default="")
    parser.add_argument("--output-json", default="reports/retrieval_ablation_research.json")
    parser.add_argument("--output-md", default="reports/retrieval_ablation_research.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    report = evaluate_retrieval_ablation(
        data_dir=data_dir,
        split=args.split,
        max_queries=args.max_queries,
        file_suffix=args.file_suffix,
    )
    write_report(report, output_json)
    output_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote retrieval ablation to {output_json}")


def evaluate_retrieval_ablation(*, data_dir: Path, split: str, max_queries: int, file_suffix: str = "") -> dict[str, object]:
    started = perf_counter()
    records_by_split = _load_records(data_dir, file_suffix=file_suffix)
    corpus = _load_corpus(data_dir, file_suffix=file_suffix)
    records = _select_records(records_by_split, split)[:max_queries]
    if not records:
        raise ValueError(f"No records found for split={split!r}")

    strategies: dict[str, tuple[object, str, str]] = {
        "bm25": (BM25Retriever(corpus), "bm25_only", "lexical"),
    }
    try:
        mini_embedder = load_embedder("sentence-transformers", "sentence-transformers/all-MiniLM-L6-v2", allow_fallback=False)
        strategies["dense_mini"] = (DenseRetriever(corpus, embedder=mini_embedder), "sentence-transformers/all-MiniLM-L6-v2", "semantic")
    except Exception as exc:
        notes.append(f"MiniLM dense retrieval unavailable: {type(exc).__name__}: {exc}")

    notes: list[str] = []
    try:
        bge_embedder = load_embedder("sentence-transformers", "BAAI/bge-m3", allow_fallback=False)
        strategies["dense_bge"] = (DenseRetriever(corpus, embedder=bge_embedder), "BAAI/bge-m3", "semantic")
    except Exception as exc:
        notes.append(f"BGE-M3 dense retrieval attempted but unavailable: {type(exc).__name__}: {exc}")

    bm25 = BM25Retriever(corpus)
    try:
        dense_mini = strategies["dense_mini"][0]
        strategies["hybrid_mini"] = (HybridRetriever(bm25, dense_mini), "bm25+all-MiniLM-L6-v2", "hybrid")
        if "dense_bge" in strategies:
            strategies["hybrid_bge"] = (HybridRetriever(bm25, strategies["dense_bge"][0]), "bm25+BGE-M3", "hybrid")
    except Exception as exc:
        notes.append(f"Hybrid retrievers could not be initialised: {type(exc).__name__}: {exc}")

    relevant_sets = [relevant_doc_ids(record) for record in records]
    strategy_metrics: dict[str, dict[str, float]] = {}
    strategy_notes: dict[str, str] = {}
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
            "memory_note": _memory_note(name, model_name),
        }

    return {
        "split": split,
        "query_count": len(records),
        "corpus_size": len(corpus),
        "runtime_seconds": round(perf_counter() - started, 3),
        "strategies": strategy_metrics,
        "notes": notes,
    }


def _load_records(data_dir: Path, *, file_suffix: str):
    if file_suffix:
        return load_records(data_dir, suffix=file_suffix)
    return load_records(data_dir)


def _load_corpus(data_dir: Path, *, file_suffix: str):
    if file_suffix:
        return load_evidence_corpus(data_dir, suffix=file_suffix)
    return load_evidence_corpus(data_dir)


def _select_records(records_by_split: dict[str, list[object]], split: str):
    selected = []
    for split_name, split_records in records_by_split.items():
        if split_name.endswith(f"_{split}"):
            selected.extend(split_records)
    return selected


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Retrieval Ablation",
        "",
        f"- split: {report['split']}",
        f"- query_count: {report['query_count']}",
        f"- corpus_size: {report['corpus_size']}",
        f"- runtime_seconds: {report['runtime_seconds']}",
        "",
        "| strategy | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | latency_seconds | backend | model_name | memory_note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for name, metrics in report["strategies"].items():
        lines.append(
            f"| {name} | {metrics['recall@1']} | {metrics['recall@5']} | {metrics['recall@10']} | {metrics['mrr']} | {metrics['ndcg@10']} | {metrics['latency_seconds']} | {metrics['backend']} | {metrics['model_name']} | {metrics['memory_note']} |"
        )
    if report.get("notes"):
        lines += ["", "## Notes", ""]
        lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


def _memory_note(strategy_name: str, model_name: str) -> str:
    if strategy_name == "bm25":
        return "Minimal incremental memory; no neural weights loaded."
    if "bge" in model_name.lower():
        return "Research model; heavier than MiniLM and may not fit free-tier memory comfortably."
    if "minilm" in model_name.lower():
        return "Moderate memory footprint (~80MB model weights plus embeddings)."
    return "Hybrid retrieval adds the dense model on top of BM25."


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
