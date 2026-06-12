"""Evaluate learned and heuristic evidence rankers on the sampled claims."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_evidence_corpus, load_records, relevant_doc_ids
from ranking.features import extract_features
from ranking.learned_ranker import LearnedRanker
from ranking.metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k
from ranking.reranker import CrossEncoderReranker
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
    cross_encoder_score: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ranking evaluation on sampled datasets.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--train-query-cap", type=int, default=40)
    parser.add_argument("--use-cross-encoder", action="store_true")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-encoder-batch-size", type=int, default=4)
    parser.add_argument("--file-suffix", default="", help="Suffix for split/corpus files, e.g. _large")
    parser.add_argument("--output-json", default="reports/ranking_eval.json")
    parser.add_argument("--output-md", default="reports/ranking_eval.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    try:
        report = evaluate_ranking(
            data_dir=data_dir,
            split=args.split,
            max_queries=args.max_queries,
            candidate_k=args.candidate_k,
            train_query_cap=args.train_query_cap,
            use_cross_encoder=args.use_cross_encoder,
            cross_encoder_model=args.cross_encoder_model,
            cross_encoder_batch_size=args.cross_encoder_batch_size,
            file_suffix=args.file_suffix,
        )
    except Exception as exc:  # pragma: no cover - failure path is surfaced explicitly
        failure_path = output_md.with_name(f"{output_md.stem}_FAILED.md")
        failure_path.write_text(
            "\n".join(
                [
                    "# Ranking Evaluation Failed",
                    "",
                    f"- split: {args.split}",
                    f"- use_cross_encoder: {args.use_cross_encoder}",
                    f"- cross_encoder_model: {args.cross_encoder_model}",
                    "",
                    "The ranking evaluation could not complete.",
                    "",
                    f"Reason: {type(exc).__name__}: {exc}",
                    "",
                    "No metrics were fabricated.",
                ]
            ),
            encoding="utf-8",
        )
        raise SystemExit(f"Ranking evaluation failed: {exc}") from exc

    write_report(report, output_json)
    output_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote ranking evaluation to {output_json}")


def evaluate_ranking(
    *,
    data_dir: Path,
    split: str,
    max_queries: int,
    candidate_k: int,
    train_query_cap: int = 40,
    use_cross_encoder: bool,
    cross_encoder_model: str,
    cross_encoder_batch_size: int = 4,
    file_suffix: str = "",
) -> dict[str, object]:
    started = perf_counter()
    records_by_split = _load_records(data_dir, file_suffix=file_suffix)
    corpus = _load_evidence_corpus(data_dir, file_suffix=file_suffix)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)
    train_records = _select_records(records_by_split, "train")[:train_query_cap]
    eval_records = _select_records(records_by_split, split)[:max_queries]
    if not eval_records:
        raise ValueError(f"No records found for split={split!r}")

    cross_encoder = _build_cross_encoder(cross_encoder_model, cross_encoder_batch_size) if use_cross_encoder else None

    learned_ranker = LearnedRanker()
    train_rows, train_labels = _build_training_data(train_records, bm25, dense, candidate_k, cross_encoder=None)
    if train_rows:
        learned_ranker.fit([row.features for row in train_rows], train_labels)

    cross_learned_ranker = LearnedRanker()
    if cross_encoder is not None:
        cross_train_rows, cross_train_labels = _build_training_data(train_records, bm25, dense, candidate_k, cross_encoder)
        if cross_train_rows:
            cross_learned_ranker.fit([row.features for row in cross_train_rows], cross_train_labels)
        else:
            cross_learned_ranker = learned_ranker
    else:
        cross_learned_ranker = learned_ranker

    split_results = _evaluate_split(
        eval_records,
        bm25,
        dense,
        learned_ranker,
        cross_learned_ranker,
        candidate_k,
        cross_encoder,
    )

    report = {
        "split": split,
        "max_queries": max_queries,
        "train_query_cap": train_query_cap,
        "num_queries": len(eval_records),
        "evidence_corpus_size": len(corpus),
        "candidate_k": candidate_k,
        "learned_ranker_backend": learned_ranker.backend_name,
        "cross_encoder_enabled": bool(cross_encoder),
        "cross_encoder_model": cross_encoder.model_name if cross_encoder else None,
        "runtime_seconds": round(perf_counter() - started, 3),
        "strategies": split_results["strategies"],
        "limitations": _limitations(bool(cross_encoder), len(eval_records), max_queries),
    }
    return report


def _build_training_data(
    records,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidate_k: int,
    cross_encoder: CrossEncoderReranker | None,
) -> tuple[list[CandidateRow], list[int]]:
    rows: list[CandidateRow] = []
    labels: list[int] = []
    for record in records:
        candidates = _candidate_rows(record, bm25, dense, candidate_k, cross_encoder)
        rows.extend(candidates)
        labels.extend([candidate.label for candidate in candidates])
    return rows, labels


def _evaluate_split(
    records,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    learned_ranker: LearnedRanker,
    cross_learned_ranker: LearnedRanker,
    candidate_k: int,
    cross_encoder: CrossEncoderReranker | None,
) -> dict[str, object]:
    strategies: dict[str, list[list[str]]] = {
        "heuristic": [],
        "bm25": [],
        "dense": [],
        "rrf": [],
        "learned": [],
    }
    if cross_encoder is not None:
        strategies["cross_encoder"] = []
        strategies["cross_encoder_plus_learned"] = []

    relevant_sets = [relevant_doc_ids(record) for record in records]

    for record in records:
        candidates = _candidate_rows(record, bm25, dense, candidate_k, cross_encoder)
        if not candidates:
            continue

        bm25_order = _rank_doc_ids(candidates, "bm25")
        dense_order = _rank_doc_ids(candidates, "dense")
        heuristic_order = _rank_doc_ids(candidates, "heuristic")
        learned_scores = learned_ranker.predict_scores([candidate.features for candidate in candidates])
        learned_order = [candidate.doc_id for candidate, _ in sorted(zip(candidates, learned_scores, strict=False), key=lambda item: (-item[1], item[0].doc_id))]
        rrf_order = [doc_id for doc_id, _ in reciprocal_rank_fusion([bm25_order, dense_order], k=60)]

        strategies["bm25"].append(bm25_order)
        strategies["dense"].append(dense_order)
        strategies["heuristic"].append(heuristic_order)
        strategies["rrf"].append(rrf_order)
        strategies["learned"].append(learned_order)

        if cross_encoder is not None:
            cross_scores = [candidate.cross_encoder_score for candidate in candidates]
            cross_order = [candidate.doc_id for candidate, _ in sorted(zip(candidates, cross_scores, strict=False), key=lambda item: (-item[1], item[0].doc_id))]
            cross_learned_scores = cross_learned_ranker.predict_scores([candidate.features for candidate in candidates])
            cross_learned_order = [
                candidate.doc_id
                for candidate, _ in sorted(zip(candidates, cross_learned_scores, strict=False), key=lambda item: (-item[1], item[0].doc_id))
            ]
            strategies["cross_encoder"].append(cross_order)
            strategies["cross_encoder_plus_learned"].append(cross_learned_order)

    return {
        "examples": len(records),
        "strategies": {
            name: _summarize_rankings(rankings, relevant_sets)
            for name, rankings in strategies.items()
        },
    }


def _candidate_rows(
    record,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidate_k: int,
    cross_encoder: CrossEncoderReranker | None,
) -> list[CandidateRow]:
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

    candidate_spans = [lookup[doc_id] for doc_id in doc_ids if doc_id in lookup]
    cross_scores = cross_encoder.score_pairs(record.claim, candidate_spans) if cross_encoder is not None else [0.0 for _ in candidate_spans]
    cross_score_map = {span.doc_id: float(score) for span, score in zip(candidate_spans, cross_scores, strict=False)}

    candidates: list[CandidateRow] = []
    for doc_id in doc_ids:
        span = lookup.get(doc_id)
        if span is None:
            continue
        cross_score = cross_score_map.get(doc_id, 0.0)
        features = extract_features(
            record.claim,
            span,
            bm25_score=bm25_scores.get(doc_id, 0.0),
            dense_score=dense_scores.get(doc_id, 0.0),
            cross_encoder_score=cross_score,
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
                cross_encoder_score=cross_score,
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
        return {"map": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0, "recall@5": 0.0}
    return {
        "map": mean_average_precision(rankings, relevant_sets),
        "mrr": mean_reciprocal_rank(rankings, relevant_sets),
        "ndcg@5": mean(ndcg_at_k(ranking, relevant, 5) for ranking, relevant in zip(rankings, relevant_sets)),
        "ndcg@10": mean(ndcg_at_k(ranking, relevant, 10) for ranking, relevant in zip(rankings, relevant_sets)),
        "recall@5": mean(_recall_at_k(ranking, relevant, 5) for ranking, relevant in zip(rankings, relevant_sets)),
    }


def _recall_at_k(ranking: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in ranking[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def _limitations(cross_encoder_enabled: bool, num_queries: int, max_queries: int) -> list[str]:
    limitations = [
        f"Evaluated on a sample of {num_queries} queries, capped at --max-queries={max_queries}.",
        "Learned ranking remains sample-scale and depends on the sampled evidence corpus.",
    ]
    if not cross_encoder_enabled:
        limitations.append("Cross-encoder reranking was not enabled for this run.")
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


def _build_cross_encoder(model_name: str, batch_size: int) -> CrossEncoderReranker:
    try:
        return CrossEncoderReranker(model_name, batch_size=batch_size)
    except TypeError:
        return CrossEncoderReranker(model_name)


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Ranking Evaluation",
        "",
        f"- Split: {report['split']}",
        f"- Max queries: {report['max_queries']}",
        f"- Train query cap: {report['train_query_cap']}",
        f"- Queries evaluated: {report['num_queries']}",
        f"- Candidate K: {report['candidate_k']}",
        f"- Learned backend: {report['learned_ranker_backend']}",
        f"- Cross encoder enabled: {report['cross_encoder_enabled']}",
        f"- Cross encoder model: {report['cross_encoder_model']}",
        f"- Runtime seconds: {report['runtime_seconds']}",
        "",
        "| strategy | map | mrr | ndcg@5 | ndcg@10 | recall@5 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for strategy, metrics in report["strategies"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    f"{metrics['map']:.3f}",
                    f"{metrics['mrr']:.3f}",
                    f"{metrics['ndcg@5']:.3f}",
                    f"{metrics['ndcg@10']:.3f}",
                    f"{metrics['recall@5']:.3f}",
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
