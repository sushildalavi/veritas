from cli import run


def test_cli_run_returns_json_payload() -> None:
    payload = run(["Paris is in France", "--top-k", "2"])

    assert payload["claim"] == "Paris is in France"
    assert payload["verdict"] in {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}
    assert "explanation" in payload
