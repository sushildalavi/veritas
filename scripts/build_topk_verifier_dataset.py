"""Build verifier datasets augmented with retrieved evidence.

For each split (train/val/test) in ``data/processed/verifier_{split}.jsonl``,
retrieve evidence with BM25 + the fine-tuned bi-encoder retriever
(``checkpoints/biencoder_retriever``), fuse with reciprocal-rank fusion, and
rerank with the fine-tuned cross-encoder reranker
(``checkpoints/cross_encoder_reranker``). Each row gets four evidence variants
formatted as ``[E1] ... [E2] ... [E3] ...``:

  - gold_evidence: the original gold evidence (oracle)
  - top1_evidence: the single highest-ranked retrieved passage
  - top3_evidence: the top-3 reranked passages
  - top5_evidence: the top-5 reranked passages
  - mixed_evidence: gold evidence for half the rows, top3 retrieved evidence
    for the other half (alternating by row index), for mixed training data

Writes:
  data/processed/verifier_{train,val,test}_topk_augmented.jsonl
  reports/topk_verifier_dataset_build.json
"""

from __future__ import annotations

import argparse
import json
from random import Random
from pathlib import Path
from time import perf_counter

from core.evidence_formatting import choose_evidence_style, render_evidence
from data.schemas import EvidenceSpan
from evaluation.reporting import write_report
from evaluation.sample_benchmarks import load_evidence_corpus, read_jsonl
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever, load_embedder
from retrieval.hybrid import HybridRetriever
from ranking.reranker import CrossEncoderReranker

SPLITS = ["train", "val", "test"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build top-k retrieved evidence augmented verifier datasets.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--evidence-corpus-suffix", default="_large")
    parser.add_argument("--biencoder-checkpoint", default="checkpoints/biencoder_retriever")
    parser.add_argument("--cross-encoder-checkpoint", default="checkpoints/cross_encoder_reranker")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=700)
    parser.add_argument("--evidence-style", choices=["plain", "passage", "evidence_letter", "bullet", "bracket", "auto"], default="auto")
    parser.add_argument("--report-json", default="reports/topk_verifier_dataset_build.json")
    return parser


def _format_evidence(spans: list[EvidenceSpan], *, style_name: str, seed_text: str) -> str:
    style = choose_evidence_style(seed_text) if style_name == "auto" else style_name
    return render_evidence(
        spans,
        style=style,
        include_title=False,
        canonicalize=False,
        shuffle=True,
        rng=Random(seed_text),
    )


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)

    started = perf_counter()
    corpus = load_evidence_corpus(data_dir, suffix=args.evidence_corpus_suffix)
    bm25 = BM25Retriever(corpus)
    embedder = load_embedder("sentence-transformers", args.biencoder_checkpoint, allow_fallback=False)
    dense = DenseRetriever(corpus, embedder=embedder)
    hybrid = HybridRetriever(bm25, dense)
    reranker = CrossEncoderReranker(model_name=args.cross_encoder_checkpoint, batch_size=8)

    split_counts: dict[str, int] = {}
    for split in SPLITS:
        rows = read_jsonl(data_dir / f"verifier_{split}.jsonl")
        if args.max_examples:
            rows = rows[: args.max_examples]
        split_counts[split] = len(rows)

        output_path = data_dir / f"verifier_{split}_topk_augmented.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                claim = str(row.get("claim", ""))
                candidates = hybrid.retrieve(claim, top_k=args.candidate_k)
                reranked = reranker.rank(claim, candidates)

                gold_evidence = str(row.get("evidence", ""))
                top1_evidence = _format_evidence(reranked[:1], style_name=args.evidence_style, seed_text=f"{claim}:top1")
                top3_evidence = _format_evidence(reranked[:3], style_name=args.evidence_style, seed_text=f"{claim}:top3")
                top5_evidence = _format_evidence(reranked[:5], style_name=args.evidence_style, seed_text=f"{claim}:top5")
                if index % 2 == 0:
                    mixed_evidence, mixed_source = gold_evidence, "gold"
                else:
                    mixed_evidence, mixed_source = top3_evidence, "retrieved_top3"

                record = {
                    "claim_id": row.get("claim_id", ""),
                    "claim": claim,
                    "label": row.get("label", ""),
                    "gold_evidence": gold_evidence,
                    "top1_evidence": top1_evidence,
                    "top3_evidence": top3_evidence,
                    "top5_evidence": top5_evidence,
                    "mixed_evidence": mixed_evidence,
                    "mixed_source": mixed_source,
                }
                handle.write(json.dumps(record) + "\n")

    report = {
        "evidence_corpus_source": f"data/processed/evidence_corpus{args.evidence_corpus_suffix}.jsonl",
        "evidence_corpus_size": len(corpus),
        "biencoder_checkpoint": args.biencoder_checkpoint,
        "cross_encoder_checkpoint": args.cross_encoder_checkpoint,
        "candidate_k": args.candidate_k,
        "max_examples_per_split": args.max_examples,
        "evidence_style": args.evidence_style,
        "split_counts": split_counts,
        "runtime_seconds": round(perf_counter() - started, 2),
        "notes": [
            "Retrieval pipeline: BM25 + fine-tuned bi-encoder (RRF fusion), reranked with the "
            "fine-tuned cross-encoder reranker.",
            "mixed_evidence alternates between gold evidence (even row index) and top3 retrieved "
            "evidence (odd row index) for mixed gold/retrieved training data.",
            f"Each split capped at {args.max_examples} examples for CPU runtime; "
            "see split_counts for actual sizes.",
        ],
    }
    write_report(report, args.report_json)
    print(f"Wrote top-k augmented verifier datasets for splits={split_counts} in {report['runtime_seconds']}s")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
