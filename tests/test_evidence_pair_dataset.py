import json
from pathlib import Path

import pytest

from data.schemas import EvidenceSpan
from retrieval.bm25 import BM25Retriever
from scripts.build_evidence_pair_dataset import build_pairs_for_split, load_jsonl

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

SPLITS = ["train", "val", "test"]


def _retriever_path(split: str) -> Path:
    return DATA_DIR / f"retriever_{split}_pairs.jsonl"


def _reranker_path(split: str) -> Path:
    return DATA_DIR / f"reranker_{split}_pairs.jsonl"


@pytest.mark.parametrize("split", SPLITS)
def test_jsonl_rows_parse(split: str) -> None:
    for path in (_retriever_path(split), _reranker_path(split)):
        rows = load_jsonl(path)
        assert rows, f"{path} should not be empty"


@pytest.mark.parametrize("split", SPLITS)
def test_no_empty_claim_or_evidence(split: str) -> None:
    for row in load_jsonl(_retriever_path(split)):
        assert row["claim"].strip()
        assert row["positive_evidence"].strip()
        for negative in row["hard_negatives"]:
            assert negative.strip()

    for row in load_jsonl(_reranker_path(split)):
        assert row["claim"].strip()
        assert row["evidence"].strip()


@pytest.mark.parametrize("split", SPLITS)
def test_no_gold_evidence_in_negatives(split: str) -> None:
    for row in load_jsonl(_retriever_path(split)):
        assert row["positive_evidence"] not in row["hard_negatives"]


@pytest.mark.parametrize("split", SPLITS)
def test_at_least_one_hard_negative_per_example(split: str) -> None:
    for row in load_jsonl(_retriever_path(split)):
        assert len(row["hard_negatives"]) >= 1


def test_label_distribution_report_exists() -> None:
    stats_path = REPORTS_DIR / "evidence_pair_data_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for split in SPLITS:
        assert "label_distribution" in stats["splits"][split]
        assert sum(stats["splits"][split]["label_distribution"].values()) > 0


def test_build_pairs_for_split_excludes_positive_from_negatives() -> None:
    corpus = [
        EvidenceSpan(doc_id="0", text="The sky is blue during the day."),
        EvidenceSpan(doc_id="1", text="Water boils at 100 degrees Celsius."),
        EvidenceSpan(doc_id="2", text="The Eiffel Tower is in Paris."),
    ]
    bm25 = BM25Retriever(passages=corpus)
    examples = [
        {
            "claim": "Water boils at a high temperature.",
            "evidence": "Water boils at 100 degrees Celsius.",
            "label": "SUPPORTED",
            "source": "fever",
        },
        {
            "claim": "Empty evidence claim.",
            "evidence": "",
            "label": "NOT_ENOUGH_INFO",
            "source": "fever",
        },
    ]

    retriever_pairs, reranker_pairs = build_pairs_for_split(
        examples=examples,
        corpus=corpus,
        bm25=bm25,
        dense_sims=None,
        bm25_top_k=3,
        num_negatives=2,
        seed=0,
    )

    assert len(retriever_pairs) == 1
    pair = retriever_pairs[0]
    assert pair["positive_evidence"] == "Water boils at 100 degrees Celsius."
    assert pair["positive_evidence"] not in pair["hard_negatives"]
    assert len(pair["hard_negatives"]) == 2

    positives = [r for r in reranker_pairs if r["relevance"] == 1]
    negatives = [r for r in reranker_pairs if r["relevance"] == 0]
    assert len(positives) == 1
    assert len(negatives) == 2
    assert all(n["negative_type"] in {"bm25_hard_negative", "dense_hard_negative", "random_negative"} for n in negatives)
