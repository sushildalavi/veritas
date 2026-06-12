from evaluation.reporting import write_report


def test_write_report_creates_json_file(tmp_path) -> None:
    path = write_report({"metric": 1.0}, tmp_path / "report.json")

    assert path.exists()
    assert '"metric": 1.0' in path.read_text(encoding="utf-8")
