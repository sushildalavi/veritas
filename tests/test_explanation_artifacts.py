from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from data.explanation_artifacts import build_dpo_pair, build_explanation_record, validate_dpo_pair, validate_explanation_record
from evaluation.sample_benchmarks import read_jsonl
from scripts.eval_explanation_model import _citation_precision, _parse_output, _to_markdown, _unsupported_rate


def test_explanation_record_schema_and_marker_leakage() -> None:
    row = read_jsonl(Path("data/processed/verifier_train.jsonl"))[0]
    record = build_explanation_record(row, split="train")

    validate_explanation_record(record)

    assert record["prompt"].startswith("You are a fact-verification assistant.")
    assert "Verifier label:" in record["prompt"]
    assert "Verifier verdict:" not in record["prompt"]
    assert record["completion"].startswith("Decision:")
    assert "Citations:" in record["completion"]
    assert "gold_label" not in record["metadata"]


def test_dpo_pair_schema_and_synthetic_rejection() -> None:
    row = read_jsonl(Path("data/processed/verifier_train.jsonl"))[0]
    record = build_explanation_record(row, split="train")
    pair = build_dpo_pair(record, rejection_type="wrong_label")

    validate_dpo_pair(pair)

    assert pair["chosen"] != pair["rejected"]
    assert pair["metadata"]["synthetic_rejection"] is True
    assert pair["metadata"]["rejection_type"] == "wrong_label"


def test_explanation_eval_parsing_helpers() -> None:
    output = "Decision: SUPPORTED\nExplanation: Grounded answer.\nCitations: [\"E1\"]"
    parsed = _parse_output(output)

    assert parsed["decision"] == "SUPPORTED"
    assert parsed["explanation"] == "Grounded answer."
    assert parsed["citations"] == ["E1"]

    row = {
        "claim": "Claim",
        "evidence_passages": [{"doc_id": "E1", "text": "Evidence text"}],
    }
    assert _citation_precision(["E1", "E2"], [type("Span", (), {"doc_id": "E1"})()]) == 0.5
    assert _unsupported_rate("Evidence text supports the claim.", row) >= 0.0


def test_explanation_eval_report_shape() -> None:
    payload = {
        "input_file": "data/explanations/sft_test.jsonl",
        "sample_size": 1,
        "backends": [
            {
                "backend": "mlx_base",
                "status": "measured",
                "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "adapter_path": None,
                "format_correctness": 0.0,
                "decision_label_consistency": 0.0,
                "citation_presence": 0.0,
                "citation_precision": 0.0,
                "unsupported_claim_rate": 0.0,
                "average_explanation_length": 0.0,
                "exact_sample_outputs": [],
            }
        ],
    }
    markdown = _to_markdown(payload)
    assert "Explanation Model Evaluation" in markdown
    assert "mlx_base" in markdown


@pytest.mark.parametrize(
    "module_path, report_prefix",
    [
        ("scripts.train_phi3_qlora", "qlora"),
        ("scripts.train_phi3_dpo", "dpo"),
    ],
)
def test_phi3_scripts_emit_skipped_reports_on_no_cuda(tmp_path: Path, monkeypatch, module_path: str, report_prefix: str) -> None:
    module = __import__(module_path, fromlist=["main"])
    config_path = tmp_path / f"{report_prefix}.yaml"
    config_path.write_text("base_model: fake-model\n", encoding="utf-8")
    report_json = tmp_path / f"{report_prefix}.json"
    report_md = tmp_path / f"{report_prefix}.md"
    before_after_md = tmp_path / f"{report_prefix}_before_after.md"

    monkeypatch.setattr(module, "_cuda_training_ready", lambda: False)
    monkeypatch.setattr(module, "_skip_reason", lambda: "CUDA is unavailable on this machine.")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            "--config",
            str(config_path),
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
            "--before-after-md",
            str(before_after_md),
        ],
    )

    with pytest.raises(SystemExit):
        module.main()

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert "CUDA is unavailable" in payload["reason"]
    assert report_md.exists()
    assert before_after_md.exists()


@pytest.mark.parametrize(
    "module_path, report_prefix",
    [
        ("scripts.train_phi3_qlora", "qlora"),
        ("scripts.train_phi3_dpo", "dpo"),
    ],
)
def test_phi3_scripts_support_dry_run(tmp_path: Path, monkeypatch, module_path: str, report_prefix: str) -> None:
    module = __import__(module_path, fromlist=["main"])
    config_path = tmp_path / f"{report_prefix}.yaml"
    config_path.write_text("base_model: fake-model\n", encoding="utf-8")
    report_json = tmp_path / f"{report_prefix}.json"
    report_md = tmp_path / f"{report_prefix}.md"
    before_after_md = tmp_path / f"{report_prefix}_before_after.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            "--dry-run",
            "--config",
            str(config_path),
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
            "--before-after-md",
            str(before_after_md),
        ],
    )

    module.main()

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run"
    assert "preflight_checks" in payload
    assert report_md.exists()
    assert before_after_md.exists()
