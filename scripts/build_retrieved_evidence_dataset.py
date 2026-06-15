"""Build a per-passage retrieved-evidence verifier dataset.

For each claim in a split, retrieves the top-k passages with the current
serving retrieval profile (`bm25_only`, per `configs/serving.yaml`) and labels
each (claim, passage) pair as ``SUPPORTS`` / ``REFUTES`` / ``NEI`` based on
whether that *individual* passage is the claim's gold evidence:

- If the passage's ``doc_id`` matches a gold-evidence doc_id of a
  SUPPORTED/REFUTED claim, the pair is a positive example
  (``pair_type=positive_retrieved``, ``pair_label`` = SUPPORTS/REFUTES,
  ``relevance=1``).
- Every SUPPORTED/REFUTED claim also gets one ``positive_oracle`` example built
  directly from its gold evidence text, so positive coverage does not depend on
  retrieval recall.
- All other retrieved passages are labeled ``NEI`` (``relevance=0``) and
  bucketed into a hard-negative type using simple, inference-time-computable
  heuristics:
    - ``same_entity_insufficient``: passage shares a title/article with the
      claim's gold evidence but is not itself gold evidence.
    - ``same_topic_missing_fact``: passage has high lexical word-overlap with
      the claim (>= 0.5 of claim content words) but is not gold evidence.
    - ``near_miss``: passage is ranked in the top-2 retrieved results but is
      not gold evidence and didn't match the above.
    - ``irrelevant``: everything else.

Each record also stores ``lexical_overlap`` (claim/passage word-overlap
fraction), usable as a relevance signal at inference time without gold labels.

Writes:
  data/processed/retrieved_evidence_verifier_{calibration,holdout}.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_project_settings
from data.schemas import ClaimEvidenceRecord, EvidenceSpan
from evaluation.sample_benchmarks import load_evidence_corpus
from scripts.eval_oracle_vs_retrieved_v2 import _load_eval_records, _normalize_label
from serving.model_loader import _load_retrieval_runtime

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "will", "with", "not", "no", "did",
    "does", "do", "than", "their", "his", "her", "they", "them", "but",
}

WORD_RE = re.compile(r"[a-z0-9]+")

SPLITS = {
    "calibration": ("fever_val", "scifact_val"),
    "holdout": ("fever_test", "scifact_test"),
}


def _tokenize(text: str) -> set[str]:
    return {token for token in WORD_RE.findall(text.lower()) if len(token) > 2 and token not in STOPWORDS}


def _lexical_overlap(claim_tokens: set[str], passage_text: str) -> float:
    if not claim_tokens:
        return 0.0
    passage_tokens = _tokenize(passage_text)
    return round(len(claim_tokens & passage_tokens) / len(claim_tokens), 4)


def _gold_evidence_text(record: ClaimEvidenceRecord) -> str | None:
    texts = [span.text.strip() for span in record.evidence if span.text and span.text.strip()]
    if not texts:
        return None
    return " ".join(texts)


def _categorize_negative(
    *,
    passage: EvidenceSpan,
    gold_titles: set[str],
    gold_label: str,
    overlap: float,
    rank: int,
) -> str:
    if gold_label != "NOT ENOUGH INFO" and passage.title and passage.title in gold_titles:
        return "same_entity_insufficient"
    if overlap >= 0.5:
        return "same_topic_missing_fact"
    if rank <= 2:
        return "near_miss"
    return "irrelevant"


def _build_pairs(record: ClaimEvidenceRecord, retrieved: list[EvidenceSpan], top_k: int) -> list[dict[str, object]]:
    gold_label = _normalize_label(record.label)
    gold_doc_ids = {span.doc_id for span in record.evidence}
    gold_titles = {span.title for span in record.evidence if span.title}
    source = str(record.metadata.get("source", "unknown"))
    claim_tokens = _tokenize(record.claim)

    pairs: list[dict[str, object]] = []

    if gold_label in {"SUPPORTED", "REFUTED"}:
        oracle_text = _gold_evidence_text(record)
        if oracle_text is not None:
            oracle_title = next((span.title for span in record.evidence if span.title), None)
            pairs.append(
                {
                    "claim_id": record.claim_id,
                    "claim": record.claim,
                    "claim_gold_label": gold_label,
                    "source": source,
                    "passage_doc_id": f"gold:{record.claim_id}",
                    "passage_title": oracle_title,
                    "passage_text": oracle_text,
                    "rank": None,
                    "retrieval_score": None,
                    "pair_label": "SUPPORTS" if gold_label == "SUPPORTED" else "REFUTES",
                    "relevance": 1,
                    "lexical_overlap": _lexical_overlap(claim_tokens, oracle_text),
                    "pair_type": "positive_oracle",
                }
            )

    for rank, passage in enumerate(retrieved[:top_k], start=1):
        overlap = _lexical_overlap(claim_tokens, passage.text)
        is_gold = passage.doc_id in gold_doc_ids
        if is_gold and gold_label in {"SUPPORTED", "REFUTED"}:
            pair_label = "SUPPORTS" if gold_label == "SUPPORTED" else "REFUTES"
            relevance = 1
            pair_type = "positive_retrieved"
        else:
            pair_label = "NEI"
            relevance = 0
            pair_type = _categorize_negative(
                passage=passage, gold_titles=gold_titles, gold_label=gold_label, overlap=overlap, rank=rank
            )

        pairs.append(
            {
                "claim_id": record.claim_id,
                "claim": record.claim,
                "claim_gold_label": gold_label,
                "source": source,
                "passage_doc_id": passage.doc_id,
                "passage_title": passage.title,
                "passage_text": passage.text,
                "rank": rank,
                "retrieval_score": passage.score,
                "pair_label": pair_label,
                "relevance": relevance,
                "lexical_overlap": overlap,
                "pair_type": pair_type,
            }
        )

    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a per-passage retrieved-evidence verifier dataset.")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--suffix", default="_large")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--output-dir", default="data/processed")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    corpus = load_evidence_corpus(data_dir, suffix=args.suffix)
    retrieval_runtime = _load_retrieval_runtime(corpus, settings)

    for split_name, prefixes in SPLITS.items():
        records = _load_eval_records(data_dir, suffix=args.suffix, split_prefixes=prefixes, max_examples=args.max_examples)
        output_path = output_dir / f"retrieved_evidence_verifier_{split_name}.jsonl"
        pair_count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                retrieved = retrieval_runtime.retriever.retrieve(record.claim, top_k=args.top_k)
                for pair in _build_pairs(record, retrieved, args.top_k):
                    handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    pair_count += 1
        print(f"{split_name}: {len(records)} claims -> {pair_count} pairs -> {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
