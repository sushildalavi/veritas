"""Evaluate learned and heuristic evidence rankers on the sampled claims."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_evidence_corpus, load_records, relevant_doc_ids
from ranking.features import extract_features
from ranking.learned_ranker import LearnedRanker
from ranking.metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import reciprocal_rank_fusion


@dataclass(frozen=True)
class CandidateRow:
    doc_id: str
    features: dict[str, float]
    label: int
    heuristic_score: float
    bm25_score: float
    dense_score: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ranking evaluation on sampled datasets.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--candidate-k", type=int, default=10)
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records_by_split = load_records(data_dir)
    corpus = load_evidence_corpus(data_dir)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)

    train_records = list(records_by_split.get("fever_train", [])) + list(records_by_split.get("scifact_train", []))
    learned_ranker = LearnedRanker()
    train_rows, train_labels = _build_training_data(train_records, bm25, dense, args.candidate_k)
    if train_rows:
        learned_ranker.fit([row.features for row in train_rows], train_labels)

    report = {
        "candidate_k": args.candidate_k,
        "splits": {},
        "ranker_backend": learned_ranker.backend_name,
    }

    for split_name, records in records_by_split.items():
        if not records:
            continue
        split_results = _evaluate_split(records, bm25, dense, learned_ranker, args.candidate_k)
        report["splits"][split_name] = split_results

    write_report(report, reports_dir / "ranking_eval.json")
    (reports_dir / "ranking_eval.md").write_text(_to_markdown(report), encoding="utf-8")
    print("Wrote ranking evaluation to reports/ranking_eval.json")


def _build_training_data(
    records,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidate_k: int,
) -> tuple[list[CandidateRow], list[int]]:
    rows: list[CandidateRow] = []
    labels: list[int] = []
    for record in records:
        candidates = _candidate_rows(record, bm25, dense, candidate_k)
        rows.extend(candidates)
        labels.extend([candidate.label for candidate in candidates])
    return rows, labels


def _evaluate_split(records, bm25: BM25Retriever, dense: DenseRetriever, learned_ranker: LearnedRanker, candidate_k: int) -> dict[str, object]:
    strategies = {"heuristic": [], "bm25": [], "dense": [], "rrf": [], "learned": []}
    relevant_sets = [relevant_doc_ids(record) for record in records]

    for record in records:
        candidates = _candidate_rows(record, bm25, dense, candidate_k)
        if not candidates:
            continue

        doc_lookup = {candidate.doc_id: candidate for candidate in candidates}
        bm25_order = _rank_doc_ids(candidates, "bm25")
        dense_order = _rank_doc_ids(candidates, "dense")
        heuristic_order = _rank_doc_ids(candidates, "heuristic")
        learned_scores = learned_ranker.predict_scores([candidate.features for candidate in candidates])
        learned_order = [candidate.doc_id for candidate, _ in sorted(zip(candidates, learned_scores), key=lambda item: (-item[1], item[0].doc_id))]
        rrf_order = [doc_id for doc_id, _ in reciprocal_rank_fusion([bm25_order, dense_order], k=60)]

        strategies["bm25"].append(bm25_order)
        strategies["dense"].append(dense_order)
        strategies["heuristic"].append(heuristic_order)
        strategies["rrf"].append(rrf_order)
        strategies["learned"].append(learned_order)

    return {
        "examples": len(records),
        "strategies": {
            name: _summarize_rankings(rankings, relevant_sets)
            for name, rankings in strategies.items()
        },
    }


def _candidate_rows(record, bm25: BM25Retriever, dense: DenseRetriever, candidate_k: int) -> list[CandidateRow]:
    bm25_results = bm25.retrieve(record.claim, top_k=candidate_k)
    dense_results = dense.retrieve(record.claim, top_k=candidate_k)
    relevant = set(relevant_doc_ids(record))

    doc_ids = []
    seen = set()
    for span in [*bm25_results, *dense_results, *[span for span in bm25.passages if span.doc_id in relevant]]:
        if span.doc_id not in seen:
            seen.add(span.doc_id)
            doc_ids.append(span.doc_id)

    bm25_positions = {span.doc_id: index + 1 for index, span in enumerate(bm25_results)}
    dense_positions = {span.doc_id: index + 1 for index, span in enumerate(dense_results)}
    bm25_scores = {span.doc_id: float(span.score or 0.0) for span in bm25_results}
    dense_scores = {span.doc_id: float(span.score or 0.0) for span in dense_results}
    lookup = {span.doc_id: span for span in [*bm25.passages, *dense.passages]}

    candidates: list[CandidateRow] = []
    for doc_id in doc_ids:
        span = lookup.get(doc_id)
        if span is None:
            continue
        features = extract_features(
            record.claim,
            span,
            bm25_score=bm25_scores.get(doc_id, 0.0),
            dense_score=dense_scores.get(doc_id, 0.0),
            bm25_rank=bm25_positions.get(doc_id),
            dense_rank=dense_positions.get(doc_id),
        )
        candidates.append(
            CandidateRow(
                doc_id=doc_id,
                features=features,
                label=1 if doc_id in relevant else 0,
                heuristic_score=features["lexical_overlap"] + features["number_overlap"] + features["date_overlap"],
                bm25_score=features["bm25_score"],
                dense_score=features["dense_similarity_score"],
            )
        )
    return candidates


def _rank_doc_ids(candidates: list[CandidateRow], strategy: str) -> list[str]:
    if strategy == "bm25":
        ordered = sorted(candidates, key=lambda item: (-item.bm25_score, item.doc_id))
    elif strategy == "dense":
        ordered = sorted(candidates, key=lambda item: (-item.dense_score, item.doc_id))
    elif strategy == "heuristic":
        ordered = sorted(candidates, key=lambda item: (-item.heuristic_score, item.doc_id))
    else:
        ordered = candidates
    return [item.doc_id for item in ordered]


def _summarize_rankings(rankings: list[list[str]], relevant_sets: list[list[str]]) -> dict[str, float]:
    if not rankings:
        return {"map": 0.0, "mrr": 0.0, "ndcg@5": 0.0}
    return {
        "map": mean_average_precision(rankings, relevant_sets),
        "mrr": mean_reciprocal_rank(rankings, relevant_sets),
        "ndcg@5": mean(ndcg_at_k(ranking, relevant, 5) for ranking, relevant in zip(rankings, relevant_sets)),
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Ranking Evaluation",
        "",
        f"- Candidate K: {report['candidate_k']}",
        f"- Learned backend: {report['ranker_backend']}",
        "",
    ]
    for split_name, split_payload in report["splits"].items():
        lines.extend([f"## {split_name}", ""])
        headers = ["strategy", "map", "mrr", "ndcg@5"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for strategy, metrics in split_payload["strategies"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        strategy,
                        f"{metrics['map']:.3f}",
                        f"{metrics['mrr']:.3f}",
                        f"{metrics['ndcg@5']:.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
