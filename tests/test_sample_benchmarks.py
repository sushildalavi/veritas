from pathlib import Path
import json

from evaluation.sample_benchmarks import build_markdown_table, load_evidence_corpus, load_records, relevant_doc_ids


def test_load_records_and_corpus(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fever_train.jsonl").write_text(
        json.dumps(
            {
                "claim_id": "1",
                "claim": "A claim",
                "label": "SUPPORTED",
                "evidence": [{"doc_id": "10", "text": "Evidence", "title": "Title"}],
                "metadata": {"source": "fever"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "evidence_corpus.jsonl").write_text(
        json.dumps({"doc_id": "10", "text": "Evidence", "title": "Title"}) + "\n",
        encoding="utf-8",
    )

    records = load_records(data_dir, ["fever_train"])
    corpus = load_evidence_corpus(data_dir)

    assert len(records["fever_train"]) == 1
    assert len(corpus) == 1
    assert relevant_doc_ids(records["fever_train"][0]) == ["10"]


def test_build_markdown_table_renders_rows() -> None:
    md = build_markdown_table([{"metric": "mrr", "value": 0.5}], ["metric", "value"])

    assert "| metric | value |" in md
    assert "mrr" in md
