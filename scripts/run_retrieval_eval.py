"""Evaluate real retrieval on the sampled FEVER and SciFact claims."""

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
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation on sampled datasets.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--dense-backend", choices=["hashing", "sentence-transformers"], default="hashing")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", default="1,5,10", help="Comma-separated retrieval cutoffs, e.g. 1,5,10")
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--title-top-k", type=int, default=0)
    parser.add_argument("--query-expansion-top-k", type=int, default=0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--include-title-in-index", action="store_true")
    parser.add_argument("--include-metadata-window", action="store_true")
    parser.add_argument("--file-suffix", default="", help="Suffix for split/corpus files, e.g. _large")
    parser.add_argument("--output-json", default="reports/retrieval_eval.json")
    parser.add_argument("--output-md", default="reports/retrieval_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    try:
        report = evaluate_retrieval(
            data_dir=data_dir,
            split=args.split,
            max_queries=args.max_queries,
            dense_backend=args.dense_backend,
            embedding_model=args.embedding_model,
            top_k=parse_top_k(args.top_k),
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            title_top_k=args.title_top_k,
            query_expansion_top_k=args.query_expansion_top_k,
            rrf_k=args.rrf_k,
            include_title_in_index=args.include_title_in_index,
            include_metadata_window=args.include_metadata_window,
            file_suffix=args.file_suffix,
        )
    except Exception as exc:  # pragma: no cover - failure path is surfaced explicitly
        failure_path = output_md.with_name(f"{output_md.stem}_FAILED.md")
        failure_path.write_text(
            "\n".join(
                [
                    "# Retrieval Evaluation Failed",
                    "",
                    f"- split: {args.split}",
                    f"- dense_backend: {args.dense_backend}",
                    f"- embedding_model: {args.embedding_model}",
                    "",
                    "The retrieval evaluation could not complete.",
                    "",
                    f"Reason: {type(exc).__name__}: {exc}",
                    "",
                    "No metrics were fabricated.",
                ]
            ),
            encoding="utf-8",
        )
        raise SystemExit(f"Retrieval evaluation failed: {exc}") from exc

    write_report(report, output_json)
    output_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote retrieval evaluation to {output_json}")


def evaluate_retrieval(
    *,
    data_dir: Path,
    split: str,
    max_queries: int,
    dense_backend: str,
    embedding_model: str,
    top_k: list[int],
    bm25_top_k: int,
    dense_top_k: int,
    title_top_k: int,
    query_expansion_top_k: int,
    rrf_k: int,
    include_title_in_index: bool,
    include_metadata_window: bool,
    file_suffix: str = "",
) -> dict[str, object]:
    started = perf_counter()
    records_by_split = _load_records(data_dir, file_suffix=file_suffix)
    corpus = _load_evidence_corpus(data_dir, file_suffix=file_suffix)
    records = _select_records(records_by_split, split)[:max_queries]
    if not records:
        raise ValueError(f"No records found for split={split!r}")

    embedder = load_embedder(
        "sentence-transformers" if dense_backend == "sentence-transformers" else "hashing",
        embedding_model if dense_backend == "sentence-transformers" else None,
        allow_fallback=False,
    )
    bm25 = BM25Retriever(
        corpus,
        include_title_in_index=include_title_in_index,
        include_metadata_window=include_metadata_window,
    )
    dense = DenseRetriever(
        corpus,
        embedder=embedder,
        include_title_in_index=include_title_in_index,
        include_metadata_window=include_metadata_window,
    )
    title_retriever = None
    if title_top_k > 0:
        title_retriever = BM25Retriever(corpus, include_title_in_index=True, include_metadata_window=include_metadata_window)
    hybrid = HybridRetriever(
        bm25,
        dense,
        title_retriever=title_retriever,
        rrf_k=rrf_k,
        bm25_top_k=bm25_top_k,
        dense_top_k=dense_top_k,
        title_top_k=title_top_k,
        query_expansion_top_k=query_expansion_top_k,
        final_top_k=max(top_k),
    )

    max_cutoff = max(top_k)
    rankings = {
        "bm25": [bm25.retrieve(record.claim, top_k=max_cutoff) for record in records],
        "dense": [dense.retrieve(record.claim, top_k=max_cutoff) for record in records],
        "hybrid": [hybrid.retrieve(record.claim, top_k=max_cutoff) for record in records],
    }
    relevant_sets = [relevant_doc_ids(record) for record in records]
    metrics = {name: _summarize(rankings[name], relevant_sets, top_k) for name in rankings}

    report = {
        "split": split,
        "max_queries": max_queries,
        "num_queries": len(records),
        "evidence_corpus_size": len(corpus),
        "dense_backend": dense.backend_name,
        "embedding_model": getattr(embedder, "model_name", embedding_model if dense_backend == "sentence-transformers" else "hashing"),
        "top_k": top_k,
        "retrieval_config": {
            "bm25_top_k": bm25_top_k,
            "dense_top_k": dense_top_k,
            "title_top_k": title_top_k,
            "query_expansion_top_k": query_expansion_top_k,
            "rrf_k": rrf_k,
            "include_title_in_index": include_title_in_index,
            "include_metadata_window": include_metadata_window,
        },
        "runtime_seconds": round(perf_counter() - started, 3),
        "metrics": metrics,
        "limitations": _limitations(dense_backend, dense.backend_name, len(records), max_queries),
    }
    return report


def _summarize(
    rankings: list[list[object]],
    relevant_sets: list[list[str]],
    top_k: list[int],
) -> dict[str, float]:
    doc_ids = [[getattr(span, "doc_id", "") for span in ranking] for ranking in rankings]
    metrics: dict[str, float] = {
        f"recall@{cutoff}": mean(recall_at_k(ids, relevant, cutoff) for ids, relevant in zip(doc_ids, relevant_sets))
        for cutoff in top_k
    }
    metrics["mrr"] = mean_reciprocal_rank(doc_ids, relevant_sets)
    for cutoff in top_k:
        metrics[f"ndcg@{cutoff}"] = mean(ndcg_at_k(ids, relevant, cutoff) for ids, relevant in zip(doc_ids, relevant_sets))
    return metrics


def _limitations(requested_backend: str, actual_backend: str, num_queries: int, max_queries: int) -> list[str]:
    limitations = [
        f"Evaluated on a sample of {num_queries} queries, capped at --max-queries={max_queries}.",
    ]
    if requested_backend == "hashing":
        limitations.append("Dense retrieval used the hashing backend, which is a lightweight baseline.")
    if requested_backend == "sentence-transformers" and actual_backend != "sentence-transformers":
        limitations.append("Requested neural backend was not available and fell back to hashing.")
    return limitations


def _select_records(records_by_split: dict[str, list[object]], split: str):
    selected = []
    for split_name, split_records in records_by_split.items():
        if split_name.endswith(f"_{split}"):
            selected.extend(split_records)
    return selected


def _load_records(data_dir: Path, *, file_suffix: str) -> dict[str, list[object]]:
    if file_suffix:
        return load_records(data_dir, suffix=file_suffix)
    return load_records(data_dir)


def _load_evidence_corpus(data_dir: Path, *, file_suffix: str):
    if file_suffix:
        return load_evidence_corpus(data_dir, suffix=file_suffix)
    return load_evidence_corpus(data_dir)


def parse_top_k(value: str) -> list[int]:
    cuts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        cuts.append(int(item))
    if not cuts:
        raise ValueError("At least one top-k cutoff is required")
    return sorted(set(cuts))


def _to_markdown(report: dict[str, object]) -> str:
    metrics = report["metrics"]
    top_k = list(report["top_k"])
    ndcg_headers = [f"ndcg@{cutoff}" for cutoff in top_k]
    metric_headers = [f"recall@{cutoff}" for cutoff in top_k] + ["mrr", *ndcg_headers]
    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Split: {report['split']}",
        f"- Max queries: {report['max_queries']}",
        f"- Queries evaluated: {report['num_queries']}",
        f"- Evidence corpus size: {report['evidence_corpus_size']}",
        f"- Dense backend: {report['dense_backend']}",
        f"- Embedding model: {report['embedding_model']}",
        f"- Runtime seconds: {report['runtime_seconds']}",
        "",
        "| retriever | " + " | ".join(metric_headers) + " |",
        "| --- | " + " | ".join(["---"] * len(metric_headers)) + " |",
    ]
    for retriever_name in ("bm25", "dense", "hybrid"):
        score = metrics[retriever_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    retriever_name,
                    *[f"{score[f'recall@{cutoff}']:.3f}" for cutoff in top_k],
                    f"{score['mrr']:.3f}",
                    *[f"{score[f'ndcg@{cutoff}']:.3f}" for cutoff in top_k],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
