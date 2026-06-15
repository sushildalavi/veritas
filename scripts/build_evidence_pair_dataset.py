"""Build retriever and reranker training pairs with hard negatives.

Reads the verifier train/val/test splits plus the large evidence corpus and produces:

- retriever_{train,val,test}_pairs.jsonl: (claim, positive_evidence, hard_negatives, label, source)
- reranker_{train,val,test}_pairs.jsonl: (claim, evidence, relevance, negative_type, label)
- evidence_pair_data_stats.{json,md}: counts and label balance for both datasets

Hard negatives come from BM25 top-k (topically similar), dense top-k
(semantically similar, via sentence-transformers if available), and random passages
from the corpus. Gold evidence text is never included in the negative set.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from data.schemas import EvidenceSpan
from retrieval.bm25 import BM25Retriever

try:  # pragma: no cover - optional dependency
    import numpy as np
    from retrieval.dense import SentenceTransformerEmbedder
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    SentenceTransformerEmbedder = None  # type: ignore


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_corpus(path: Path) -> list[EvidenceSpan]:
    passages = []
    for row in load_jsonl(path):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        passages.append(
            EvidenceSpan(
                doc_id=str(row.get("doc_id", "")),
                text=text,
                title=row.get("title"),
                metadata={"split": row.get("split")},
            )
        )
    return passages


def load_dense_embedder() -> "SentenceTransformerEmbedder | None":
    if SentenceTransformerEmbedder is None:
        return None
    try:
        return SentenceTransformerEmbedder()
    except Exception:  # pragma: no cover - optional dependency
        return None


def eligible_examples(examples: list[dict]) -> list[dict]:
    """Examples with a usable claim and non-empty evidence text."""

    eligible = []
    for ex in examples:
        claim = (ex.get("claim") or "").strip()
        evidence = (ex.get("evidence") or "").strip()
        if claim and evidence:
            eligible.append(ex)
    return eligible


def mine_hard_negatives(
    *,
    claim: str,
    positive_evidence: str,
    bm25: BM25Retriever,
    bm25_top_k: int,
    dense_row: "np.ndarray | None",
    corpus: list[EvidenceSpan],
    rng: random.Random,
    bm25_negatives: int,
    dense_negatives: int,
    random_negatives: int,
) -> list[tuple[str, str]]:
    """Return source-balanced (text, negative_type) pairs, excluding the positive."""

    seen: set[str] = {positive_evidence}
    negatives: list[tuple[str, str]] = []
    negatives.extend(
        _collect_bm25_negatives(
            claim=claim,
            bm25=bm25,
            top_k=bm25_top_k,
            limit=bm25_negatives,
            seen=seen,
        )
    )
    negatives.extend(
        _collect_dense_negatives(
            dense_row=dense_row,
            corpus=corpus,
            limit=dense_negatives,
            search_depth=bm25_top_k * 3,
            seen=seen,
        )
    )
    negatives.extend(
        _collect_random_negatives(
            corpus=corpus,
            rng=rng,
            limit=random_negatives,
            seen=seen,
        )
    )
    if len(negatives) < bm25_negatives + dense_negatives + random_negatives:
        negatives.extend(
            _collect_random_negatives(
                corpus=corpus,
                rng=rng,
                limit=(bm25_negatives + dense_negatives + random_negatives) - len(negatives),
                seen=seen,
            )
        )

    return negatives


def _collect_bm25_negatives(
    *,
    claim: str,
    bm25: BM25Retriever,
    top_k: int,
    limit: int,
    seen: set[str],
) -> list[tuple[str, str]]:
    negatives: list[tuple[str, str]] = []
    for span in bm25.retrieve(claim, top_k=top_k):
        if span.text in seen:
            continue
        seen.add(span.text)
        negatives.append((span.text, "bm25_hard_negative"))
        if len(negatives) >= limit:
            break
    return negatives


def _collect_dense_negatives(
    *,
    dense_row: "np.ndarray | None",
    corpus: list[EvidenceSpan],
    limit: int,
    search_depth: int,
    seen: set[str],
) -> list[tuple[str, str]]:
    negatives: list[tuple[str, str]] = []
    if dense_row is None:
        return negatives
    ranked = sorted(range(len(corpus)), key=lambda i: -float(dense_row[i]))
    for index in ranked[:search_depth]:
        text = corpus[index].text
        if text in seen:
            continue
        seen.add(text)
        negatives.append((text, "dense_hard_negative"))
        if len(negatives) >= limit:
            break
    return negatives


def _collect_random_negatives(
    *,
    corpus: list[EvidenceSpan],
    rng: random.Random,
    limit: int,
    seen: set[str],
) -> list[tuple[str, str]]:
    negatives: list[tuple[str, str]] = []
    attempts = 0
    while len(negatives) < limit and attempts < max(20, limit * 20):
        attempts += 1
        candidate = corpus[rng.randrange(len(corpus))].text
        if candidate in seen:
            continue
        seen.add(candidate)
        negatives.append((candidate, "random_negative"))
    return negatives


def build_pairs_for_split(
    *,
    examples: list[dict],
    corpus: list[EvidenceSpan],
    bm25: BM25Retriever,
    dense_sims: "np.ndarray | None",
    bm25_top_k: int,
    bm25_negatives: int,
    dense_negatives: int,
    random_negatives: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    retriever_pairs: list[dict] = []
    reranker_pairs: list[dict] = []

    for row_index, ex in enumerate(eligible_examples(examples)):
        claim = ex["claim"].strip()
        positive_evidence = ex["evidence"].strip()
        label = ex["label"]
        source = ex.get("source", "")

        dense_row = dense_sims[row_index] if dense_sims is not None else None
        negatives = mine_hard_negatives(
            claim=claim,
            positive_evidence=positive_evidence,
            bm25=bm25,
            bm25_top_k=bm25_top_k,
            dense_row=dense_row,
            corpus=corpus,
            rng=rng,
            bm25_negatives=bm25_negatives,
            dense_negatives=dense_negatives,
            random_negatives=random_negatives,
        )

        retriever_pairs.append(
            {
                "claim": claim,
                "positive_evidence": positive_evidence,
                "hard_negatives": [text for text, _ in negatives],
                "label": label,
                "source": source,
            }
        )

        reranker_pairs.append(
            {
                "claim": claim,
                "evidence": positive_evidence,
                "relevance": 1,
                "negative_type": "gold",
                "label": label,
            }
        )
        for text, negative_type in negatives:
            reranker_pairs.append(
                {
                    "claim": claim,
                    "evidence": text,
                    "relevance": 0,
                    "negative_type": negative_type,
                    "label": label,
                }
            )

    return retriever_pairs, reranker_pairs


def compute_dense_sims(
    embedder: "SentenceTransformerEmbedder | None",
    corpus: list[EvidenceSpan],
    examples: list[dict],
) -> "np.ndarray | None":
    if embedder is None or np is None:
        return None
    eligible = eligible_examples(examples)
    if not eligible:
        return None
    corpus_emb = np.asarray(embedder.encode([span.text for span in corpus]), dtype=np.float32)
    query_emb = np.asarray(embedder.encode([ex["claim"].strip() for ex in eligible]), dtype=np.float32)
    return query_emb @ corpus_emb.T


def label_distribution(rows: list[dict], key: str = "label") -> dict[str, int]:
    return dict(Counter(row[key] for row in rows))


def build_stats(
    splits: dict[str, tuple[list[dict], list[dict]]],
    corpus_size: int,
) -> dict:
    stats: dict = {"corpus_size": corpus_size, "splits": {}}
    for split_name, (retriever_pairs, reranker_pairs) in splits.items():
        positives = sum(1 for r in reranker_pairs if r["relevance"] == 1)
        negatives = sum(1 for r in reranker_pairs if r["relevance"] == 0)
        negative_types = Counter(r["negative_type"] for r in reranker_pairs if r["relevance"] == 0)
        hard_negative_counts = Counter(len(p["hard_negatives"]) for p in retriever_pairs)
        stats["splits"][split_name] = {
            "positive_examples": positives,
            "negative_examples": negatives,
            "retriever_pairs": len(retriever_pairs),
            "reranker_pairs": len(reranker_pairs),
            "label_distribution": label_distribution(retriever_pairs),
            "reranker_relevance_counts": {"positive": positives, "negative": negatives},
            "reranker_negative_type_counts": dict(negative_types),
            "hard_negative_count_distribution": {str(k): v for k, v in sorted(hard_negative_counts.items())},
        }
    return stats


def build_markdown(stats: dict) -> str:
    lines = ["# Evidence Pair Dataset Stats", ""]
    lines.append(f"- Corpus size: {stats['corpus_size']}")
    lines.append("")
    for split_name, split_stats in stats["splits"].items():
        lines.append(f"## {split_name}")
        lines.append("")
        lines.append(f"- Positive examples: {split_stats['positive_examples']}")
        lines.append(f"- Negative examples: {split_stats['negative_examples']}")
        lines.append(f"- Retriever pairs: {split_stats['retriever_pairs']}")
        lines.append(f"- Reranker pairs: {split_stats['reranker_pairs']}")
        lines.append(f"- Reranker relevance counts: {split_stats['reranker_relevance_counts']}")
        lines.append(f"- Reranker negative type counts: {split_stats['reranker_negative_type_counts']}")
        lines.append(f"- Label distribution (retriever pairs): {split_stats['label_distribution']}")
        lines.append(f"- Hard negative count distribution: {split_stats['hard_negative_count_distribution']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/verifier_train.jsonl")
    parser.add_argument("--val", default="data/processed/verifier_val.jsonl")
    parser.add_argument("--test", default="data/processed/verifier_test.jsonl")
    parser.add_argument("--corpus", default="data/processed/evidence_corpus_large.jsonl")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--bm25-top-k", type=int, default=8)
    parser.add_argument("--bm25-hard-negatives", type=int, default=2)
    parser.add_argument("--dense-hard-negatives", type=int, default=2)
    parser.add_argument("--random-negatives", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dense", action="store_true", help="Skip dense hard-negative mining")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(Path(args.corpus))
    bm25 = BM25Retriever(passages=corpus)
    embedder = None if args.no_dense else load_dense_embedder()

    split_examples = {
        "train": load_jsonl(Path(args.train)),
        "val": load_jsonl(Path(args.val)),
        "test": load_jsonl(Path(args.test)),
    }

    results: dict[str, tuple[list[dict], list[dict]]] = {}
    for split_name, examples in split_examples.items():
        dense_sims = compute_dense_sims(embedder, corpus, examples)
        retriever_pairs, reranker_pairs = build_pairs_for_split(
            examples=examples,
            corpus=corpus,
            bm25=bm25,
            dense_sims=dense_sims,
            bm25_top_k=args.bm25_top_k,
            bm25_negatives=args.bm25_hard_negatives,
            dense_negatives=0 if args.no_dense else args.dense_hard_negatives,
            random_negatives=args.random_negatives,
            seed=args.seed,
        )
        results[split_name] = (retriever_pairs, reranker_pairs)

        write_jsonl(output_dir / f"retriever_{split_name}_pairs.jsonl", retriever_pairs)
        write_jsonl(output_dir / f"reranker_{split_name}_pairs.jsonl", reranker_pairs)
        print(
            f"{split_name}: {len(retriever_pairs)} retriever pairs, "
            f"{len(reranker_pairs)} reranker pairs"
        )

    stats = build_stats(results, corpus_size=len(corpus))
    stats["dense_backend"] = getattr(embedder, "backend_name", "none") if embedder else "none"

    (reports_dir / "evidence_pair_data_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    (reports_dir / "evidence_pair_data_stats.md").write_text(build_markdown(stats), encoding="utf-8")
    print(f"Wrote {reports_dir}/evidence_pair_data_stats.json and {reports_dir}/evidence_pair_data_stats.md")


if __name__ == "__main__":
    main()
