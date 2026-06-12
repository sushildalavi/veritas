from data.export_demo_corpus import export_demo_corpus


def test_export_demo_corpus_writes_jsonl(tmp_path) -> None:
    output = export_demo_corpus(tmp_path / "demo.jsonl")

    text = output.read_text(encoding="utf-8")
    assert text.count("\n") >= 2
    assert '"title": "demo-0"' in text
