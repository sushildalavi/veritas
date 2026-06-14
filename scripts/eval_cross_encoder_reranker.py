"""Evaluate the fine-tuned cross-encoder reranker against the generic MS MARCO model.

Runs the existing ranking evaluation (``scripts.run_ranking_eval.evaluate_ranking``) twice
on the same query set, once with the generic ``cross-encoder/ms-marco-MiniLM-L-6-v2``
checkpoint and once with the fine-tuned checkpoint produced by
``scripts/train_cross_encoder_reranker.py``, and compares:

  - rrf: BM25 + dense reciprocal-rank fusion (no reranker)
  - learned: sklearn/lightgbm learned ranker over heuristic features
  - cross_encoder_generic: off-the-shelf cross-encoder/ms-marco-MiniLM-L-6-v2
  - cross_encoder_finetuned: cross-encoder fine-tuned on reranker pairs
  - cross_encoder_finetuned_plus_learned: fine-tuned cross-encoder score added to the
    learned-ranker feature set

Writes:
  reports/cross_encoder_reranker_eval.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from evaluation.reporting import write_report
from run_ranking_eval import evaluate_ranking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned cross-encoder reranker.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--train-query-cap", type=int, default=40)
    parser.add_argument("--file-suffix", default="_large")
    parser.add_argument("--generic-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--checkpoint", default="checkpoints/cross_encoder_reranker")
    parser.add_argument("--cross-encoder-batch-size", type=int, default=8)
    parser.add_argument("--output-json", default="reports/cross_encoder_reranker_eval.json")
    parser.add_argument("--output-md", default="reports/cross_encoder_reranker_eval.md")
    return parser


def evaluate_cross_encoder_reranker(
    *,
    data_dir: Path,
    split: str,
    max_queries: int,
    candidate_k: int,
    train_query_cap: int,
    file_suffix: str,
    generic_model: str,
    checkpoint: str,
    cross_encoder_batch_size: int,
) -> dict[str, object]:
    started = perf_counter()
    notes: list[str] = []

    generic_report = evaluate_ranking(
        data_dir=data_dir,
        split=split,
        max_queries=max_queries,
        candidate_k=candidate_k,
        train_query_cap=train_query_cap,
        use_cross_encoder=True,
        cross_encoder_model=generic_model,
        cross_encoder_batch_size=cross_encoder_batch_size,
        file_suffix=file_suffix,
    )

    strategies: dict[str, dict[str, float]] = {
        "rrf": generic_report["strategies"]["rrf"],
        "learned": generic_report["strategies"]["learned"],
        "cross_encoder_generic": generic_report["strategies"]["cross_encoder"],
    }

    try:
        finetuned_report = evaluate_ranking(
            data_dir=data_dir,
            split=split,
            max_queries=max_queries,
            candidate_k=candidate_k,
            train_query_cap=train_query_cap,
            use_cross_encoder=True,
            cross_encoder_model=checkpoint,
            cross_encoder_batch_size=cross_encoder_batch_size,
            file_suffix=file_suffix,
        )
        strategies["cross_encoder_finetuned"] = finetuned_report["strategies"]["cross_encoder"]
        strategies["cross_encoder_finetuned_plus_learned"] = finetuned_report["strategies"]["cross_encoder_plus_learned"]
    except Exception as exc:
        notes.append(f"Fine-tuned cross-encoder unavailable: {type(exc).__name__}: {exc}")

    if "cross_encoder_finetuned" in strategies:
        generic = strategies["cross_encoder_generic"]
        finetuned = strategies["cross_encoder_finetuned"]
        for metric in ("map", "mrr", "ndcg@5", "ndcg@10", "recall@5"):
            delta = round(float(finetuned[metric]) - float(generic[metric]), 4)
            direction = "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged")
            notes.append(f"Fine-tuned vs generic cross-encoder {metric}: {direction} by {abs(delta)} ({generic[metric]} -> {finetuned[metric]}).")

    return {
        "split": split,
        "num_queries": generic_report["num_queries"],
        "candidate_k": candidate_k,
        "evidence_corpus_size": generic_report["evidence_corpus_size"],
        "generic_model": generic_model,
        "checkpoint": checkpoint,
        "runtime_seconds": round(perf_counter() - started, 3),
        "strategies": strategies,
        "notes": notes,
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Cross-Encoder Reranker Evaluation",
        "",
        f"- split: {report['split']}",
        f"- num_queries: {report['num_queries']}",
        f"- candidate_k: {report['candidate_k']}",
        f"- evidence_corpus_size: {report['evidence_corpus_size']}",
        f"- generic_model: {report['generic_model']}",
        f"- checkpoint: {report['checkpoint']}",
        f"- runtime_seconds: {report['runtime_seconds']}",
        "",
        "| strategy | map | mrr | ndcg@5 | ndcg@10 | recall@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["strategies"].items():
        lines.append(
            f"| {name} | {round(metrics['map'], 4)} | {round(metrics['mrr'], 4)} | "
            f"{round(metrics['ndcg@5'], 4)} | {round(metrics['ndcg@10'], 4)} | {round(metrics['recall@5'], 4)} |"
        )
    if report.get("notes"):
        lines += ["", "## Notes", ""]
        lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    report = evaluate_cross_encoder_reranker(
        data_dir=Path(args.data_dir),
        split=args.split,
        max_queries=args.max_queries,
        candidate_k=args.candidate_k,
        train_query_cap=args.train_query_cap,
        file_suffix=args.file_suffix,
        generic_model=args.generic_model,
        checkpoint=args.checkpoint,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    write_report(report, output_json)
    output_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote cross-encoder reranker eval to {output_json}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
