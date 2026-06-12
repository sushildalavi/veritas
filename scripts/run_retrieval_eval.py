"""Evaluate real retrieval on the sampled FEVER and SciFact claims."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from data.sample_pipeline import build_data_quality_markdown
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_evidence_corpus, load_records, relevant_doc_ids
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation on sampled datasets.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records_by_split = load_records(data_dir)
    corpus = load_evidence_corpus(data_dir)

    retrievers = {
        "bm25": BM25Retriever(corpus),
        "dense": DenseRetriever(corpus),
    }
    retrievers["hybrid"] = HybridRetriever(retrievers["bm25"], retrievers["dense"])

    report = {
        "top_k": args.top_k,
        "corpus_size": len(corpus),
        "splits": {},
    }

    for split_name, records in records_by_split.items():
        if not records:
            continue
        report["splits"][split_name] = {}
        for retriever_name, retriever in retrievers.items():
            rankings = [retriever.retrieve(record.claim, top_k=args.top_k) for record in records]
            retrieved_ids = [[span.doc_id for span in ranking] for ranking in rankings]
            relevant_sets = [relevant_doc_ids(record) for record in records]
            report["splits"][split_name][retriever_name] = {
                "mean_recall@1": mean(recall_at_k(ids, relevant, 1) for ids, relevant in zip(retrieved_ids, relevant_sets)),
                "mean_recall@5": mean(recall_at_k(ids, relevant, min(5, args.top_k)) for ids, relevant in zip(retrieved_ids, relevant_sets)),
                "mean_recall@10": mean(recall_at_k(ids, relevant, min(10, args.top_k)) for ids, relevant in zip(retrieved_ids, relevant_sets)),
                "mean_mrr": mean_reciprocal_rank(retrieved_ids, relevant_sets),
                "mean_ndcg@5": mean(ndcg_at_k(ids, relevant, min(5, args.top_k)) for ids, relevant in zip(retrieved_ids, relevant_sets)),
            }

    write_report(report, reports_dir / "retrieval_eval.json")
    (reports_dir / "retrieval_eval.md").write_text(_to_markdown(report), encoding="utf-8")
    print("Wrote retrieval evaluation to reports/retrieval_eval.json")


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Corpus size: {report['corpus_size']}",
        f"- Top K: {report['top_k']}",
        "",
    ]
    for split_name, split_report in report["splits"].items():
        lines.extend([f"## {split_name}", ""])
        headers = ["retriever", "mean_recall@1", "mean_recall@5", "mean_recall@10", "mean_mrr", "mean_ndcg@5"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for retriever_name, metrics in split_report.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        retriever_name,
                        f"{metrics['mean_recall@1']:.3f}",
                        f"{metrics['mean_recall@5']:.3f}",
                        f"{metrics['mean_recall@10']:.3f}",
                        f"{metrics['mean_mrr']:.3f}",
                        f"{metrics['mean_ndcg@5']:.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
